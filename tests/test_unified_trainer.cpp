// L073: UnifiedTrainer entry contract (Phase 16 Wave 7).
//
// Covers the single-entry facade without running a training loop: kind
// parsing/naming, config translation, and configure() validation. The
// delegated fit/step paths are the pre-existing Trainer::fit/micro_step
// implementation, covered by test_trainer/test_training/test_gradient_check.
#include "quant/trainer.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/tokenizer.h"
#include "quant/test.h"

#include <cstdio>
#include <string>

using namespace quant;

static void test_kind_names() {
    TEST_SUITE("L073: kind names");
    TEST_CHECK(std::string(unified_trainer_kind_name(UnifiedTrainerKind::Dense)) == "dense",
               "dense name");
    TEST_CHECK(std::string(unified_trainer_kind_name(UnifiedTrainerKind::FineTune)) == "finetune",
               "finetune name");
    TEST_CHECK(std::string(unified_trainer_kind_name(UnifiedTrainerKind::MoE)) == "moe",
               "moe name");
    TEST_CHECK(std::string(unified_trainer_kind_name(UnifiedTrainerKind::Native)) == "native",
               "native name");
}

static void test_parse_kind() {
    TEST_SUITE("L073: parse kind");
    UnifiedTrainerKind k = UnifiedTrainerKind::Dense;
    TEST_CHECK(unified_trainer_parse_kind("dense", &k) && k == UnifiedTrainerKind::Dense,
               "parse dense");
    TEST_CHECK(unified_trainer_parse_kind("finetune", &k) && k == UnifiedTrainerKind::FineTune,
               "parse finetune");
    TEST_CHECK(unified_trainer_parse_kind("FT", &k) && k == UnifiedTrainerKind::FineTune,
               "parse FT alias (case-insensitive)");
    TEST_CHECK(unified_trainer_parse_kind("moe", &k) && k == UnifiedTrainerKind::MoE,
               "parse moe");
    TEST_CHECK(!unified_trainer_parse_kind("bogus", &k), "bogus kind rejected");
    TEST_CHECK(!unified_trainer_parse_kind("", nullptr), "empty kind rejected");
}

static void test_config_translation() {
    TEST_SUITE("L073: config translation");
    UnifiedTrainArgs a;
    a.batch_size = 4;
    a.seq_length = 64;
    a.num_epochs = 2;
    a.learning_rate = 1e-4f;
    a.output_path = "u.quant";
    a.use_qat = true;
    a.qat_bits = 4;
    TrainConfig c = unified_to_train_config(a);
    TEST_CHECK(c.batch_size == 4 && c.seq_length == 64, "batch/seq carried");
    TEST_CHECK(c.num_epochs == 2, "epochs carried");
    TEST_CHECK(c.learning_rate == 1e-4f, "lr carried");
    TEST_CHECK(c.output_path == "u.quant", "output carried");
    TEST_CHECK(c.use_qat && c.qat_bits == 4, "qat carried");
}

static TransformerConfig tiny_cfg() {
    TransformerConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.num_kv_heads = 2;
    cfg.max_seq_len = 32;
    return cfg;
}

static void test_configure_rejects() {
    TEST_SUITE("L073: configure validation");
    TransformerConfig tc = tiny_cfg();
    DenseModel m(tc);
    BPETokenizer tok;
    std::string err;

    UnifiedTrainer t_null(nullptr, nullptr);
    UnifiedTrainArgs a;
    TEST_CHECK(!t_null.configure(a, &err) && !err.empty(), "null model/tokenizer rejected");

    UnifiedTrainer t_bad(&m, &tok);
    UnifiedTrainArgs bad = a;
    bad.batch_size = 0;
    TEST_CHECK(!t_bad.configure(bad, &err), "zero batch rejected");
    bad = a;
    bad.optimizer_name = "sgd";
    TEST_CHECK(!t_bad.configure(bad, &err), "unknown optimizer rejected");
    bad = a;
    bad.num_epochs = 0;
    TEST_CHECK(!t_bad.configure(bad, &err), "zero epochs rejected");
}

static void test_configure_ok() {
    TEST_SUITE("L073: configure success");
    TransformerConfig tc = tiny_cfg();
    DenseModel m(tc);
    BPETokenizer tok;
    UnifiedTrainArgs a;
    a.kind = UnifiedTrainerKind::FineTune;
    a.batch_size = 2;
    a.seq_length = 8;
    a.num_epochs = 1;
    a.optimizer_name = "adamw";
    a.output_path = "_test_unified.quant";
    UnifiedTrainer t(&m, &tok);
    std::string err;
    TEST_CHECK(t.configure(a, &err), "valid args configure");
    TEST_CHECK(t.inner() != nullptr, "inner trainer built");
    TEST_CHECK(t.args().batch_size == 2, "args echoed");
    TEST_CHECK(std::string(unified_trainer_kind_name(t.args().kind)) == "finetune",
               "kind echoed");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Transcender - Unified Trainer Entry (L073) Suite\n");
    printf("================================================\n");

    test_kind_names();
    test_parse_kind();
    test_config_translation();
    test_configure_rejects();
    test_configure_ok();

    printf("\n================================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
