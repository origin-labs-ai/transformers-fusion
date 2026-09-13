// test_agi_flywheel.cpp — Flywheel construction, history, log path
#include "quant/agi.h"
#include "quant/transformer.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/test.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace quant;

int main() {
    TEST_SUITE("agi_flywheel");
    printf("=== AGI flywheel test ===\n\n");

    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    Trainer trainer(&model, nullptr);
    agi::SafetyGuardrails safety;
    agi::CodeGenSelfImprover codegen(&model);
    agi::SelfVerifier verifier(&model);
    agi::CapabilityAmplifier amplifier(&model);

    // Construct flywheel with all subsystems
    agi::Flywheel flywheel(&model, &trainer, &codegen, &verifier, &amplifier, &safety);

    TEST_CHECK(flywheel.get_history().empty(), "flywheel history empty at start");
    TEST_CHECK(flywheel.get_no_improvement_count() == 0, "no improvements at start");

    std::string log_path = flywheel.get_log_path();
    TEST_CHECK(!log_path.empty(), "flywheel log path non-empty");
    printf("  flywheel log path: %s\n", log_path.c_str());

    // Subsystem verification (used inside the loop)
    TEST_CHECK(codegen.compile_and_test("int main(){return 0;}\n"),
               "codegen compiles trivial program");

    // P17: real asserts for self_play (no pass-by-construction).
    // Null model skips the model-override branch -> deterministic template round-robin.
    agi::Flywheel fw_null(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    std::string t1 = fw_null.self_play_for_test();
    std::string t2 = fw_null.self_play_for_test();
    TEST_CHECK(!t1.empty(), "self_play returns non-empty");
    TEST_CHECK(!t2.empty(), "self_play second call non-empty");
    TEST_CHECK(t1 != t2, "task_index round-robins (consecutive tasks differ)");

    // P17: dataset-driven load via AGI_TASKS_FILE (file path, not hardcoded).
    {
        const char* tmp_tasks = "flywheel_p17_test_tasks.txt";
        {
            std::ofstream ofs(tmp_tasks);
            ofs << "# P17 test task list\n\nCustom task alpha P17\nCustom task beta P17\n";
        }
#ifdef _WIN32
        _putenv("AGI_TASKS_FILE=flywheel_p17_test_tasks.txt");
#else
        setenv("AGI_TASKS_FILE", "flywheel_p17_test_tasks.txt", 1);
#endif
        agi::Flywheel fw_file(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        std::string ft = fw_file.self_play_for_test();
        TEST_CHECK(ft == "Custom task alpha P17" || ft == "Custom task beta P17",
                   "AGI_TASKS_FILE dataset-driven load");
        std::string ft2 = fw_file.self_play_for_test();
        TEST_CHECK(!ft2.empty(), "file-backed self_play non-empty");
#ifdef _WIN32
        _putenv("AGI_TASKS_FILE=");
#else
        unsetenv("AGI_TASKS_FILE");
#endif
        std::remove(tmp_tasks);
    }

    int failures = TEST_REPORT();
    printf("\nAGI FLYWHEEL TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
