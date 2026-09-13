// L079: missing-module smoke tests (Phase 16 Wave 7) + L072 evidence suite.
//
// Covers: quant::WorldModel, quant::agi::WorldModel, quant::multi_agent,
// OCR/Video/Audio header contracts, UnigramTokenizer SentencePiece TSV.
// SCOPE NOTE (honest): quant::OCREngine / VideoEncoder / AudioEncoder /
// MelSpectrogram / AudioFeatureExtractor functional methods (init/encode/
// extract/recognize) are DECLARED in include/quant/{ocr,video,audio}.h but
// DEFINED nowhere (same for Image(w,h,c)/to_tensor/load, Video::load).
// Calling them would fail at LINK, so this suite covers what exists and is
// link-safe (construction, defaults, inline accessors). Functional coverage
// is retired until the implementations land; the link gap is reported, not
// papered over.
#include "quant/agi.h"
#include "quant/world_model.h"
#include "quant/multi_agent.h"
#include "quant/ocr.h"
#include "quant/video.h"
#include "quant/audio.h"
#include "quant/image.h"
#include "quant/bpe_tokenizer.h"
#include "quant/tensor.h"
#include "quant/test.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace quant;

static void test_world_model_legacy() {
    TEST_SUITE("L079: quant::WorldModel (null-safe smoke)");
    quant::WorldModel wm(nullptr);
    Tensor state({4});
    state.zero_();
    Tensor action({4});
    action.zero_();
    Tensor next = wm.simulate_step(state, action);
    TEST_CHECK(next.numel() == 4, "simulate_step preserves numel without model");
    const float* d = next.data<float>();
    bool all_zero = true;
    for (int i = 0; i < 4; i++) all_zero = all_zero && (d[i] == 0.0f);
    TEST_CHECK(all_zero, "simulate_step returns zeros without model");
    auto p = wm.plan(3);
    TEST_CHECK(p.empty(), "plan returns empty without model");
}

static void test_world_model_agi() {
    TEST_SUITE("L079: quant::agi::WorldModel (null-model smoke)");
    agi::WorldModel wm(nullptr, 8, 2);
    agi::WorldState s;
    s.set_value("x", 1.0f);
    s.set_value("y", 2.0f);
    TEST_CHECK(s.get_value("x") == 1.0f, "WorldState set/get roundtrip");
    TEST_CHECK(s.get_value("missing") == 0.0f, "WorldState missing key is 0");

    agi::PredictionResult r = wm.predict(s, "move_forward");
    TEST_CHECK(r.predicted_state.timestamp == s.timestamp + 1, "predict advances timestamp");
    TEST_CHECK(r.uncertainty == 1.0f, "null-model uncertainty is 1");
    TEST_CHECK(r.confidence == 0.0f, "null-model confidence is 0");
    TEST_CHECK(wm.predict_reward(s, "wait") == 0.0f, "null-model reward is 0");

    wm.store_state(s, 1.0f);
    agi::WorldState back = wm.retrieve_most_relevant("x");
    TEST_CHECK(back.get_value("x") == 1.0f, "state memory retrieve roundtrip");
    agi::WorldModelStats st = wm.get_stats();
    TEST_CHECK(st.total_predictions == 0, "no predictions counted yet");
    TEST_CHECK(st.memory_entries >= 1, "memory holds stored state");

    wm.set_current_state(s);
    TEST_CHECK(wm.get_current_state().get_value("y") == 2.0f, "current state roundtrip");

    agi::Transition t;
    t.state = s;
    t.action = "wait";
    t.next_state = s;
    wm.add_transition(t);
    TEST_CHECK(!wm.get_transition_buffer().empty(), "transition buffer grows");
}

static void test_multi_agent() {
    TEST_SUITE("L079: MultiAgentSystem smoke");
    multi_agent::MultiAgentSystem sys(1);
    TEST_CHECK(sys.agent_count() == 1, "ctor pre-creates 1 agent");
    int coder = sys.add_agent(multi_agent::AgentRole::CODER, "coder");
    int tester = sys.add_agent(multi_agent::AgentRole::TESTER, "tester");
    TEST_CHECK(sys.agent_count() == 3, "add_agent grows roster");
    TEST_CHECK(sys.get_agent(coder).name == "coder", "agent name kept");

    multi_agent::AgentMessage m;
    m.from_agent = coder;
    m.to_agent = tester;
    m.type = multi_agent::MessageType::TASK_ASSIGN;
    m.content = "write test";
    sys.send_message(m);
    auto inbox = sys.receive_messages(tester);
    TEST_CHECK(!inbox.empty(), "direct message delivered");
    auto bc = sys.broadcast(coder, multi_agent::MessageType::STATUS_UPDATE, "hi");
    TEST_CHECK(!bc.empty(), "broadcast reaches agents");

    sys.assign_task(coder, "implement feature");
    sys.decompose_task("big task", 3);
    TEST_CHECK(!sys.get_subtasks().empty(), "decompose creates subtasks");

    sys.submit_vote(coder, "plan-A", true, 0.9f, "looks good");
    sys.submit_vote(tester, "plan-A", true, 0.7f, "fine");
    TEST_CHECK(sys.check_consensus(0.6f), "unanimous votes reach consensus");
    TEST_CHECK(sys.get_votes().size() == 2, "votes recorded");

    sys.write_blackboard("goal", "ship", coder);
    TEST_CHECK(sys.read_blackboard("goal") == "ship", "blackboard roundtrip");
    TEST_CHECK(!sys.search_blackboard("go").empty(), "blackboard prefix search");

    auto plan = sys.create_plan("release", 2);
    TEST_CHECK(!plan.high_level_steps.empty(), "plan has steps");
    sys.run_round(1);
    TEST_CHECK(true, "run_round completes without crash");
    TEST_CHECK(!sys.metrics_summary().empty(), "metrics summary non-empty");
}

static void test_ocr_video_audio_contract() {
    TEST_SUITE("L079: OCR/Video/Audio header contract (link-safe subset)");
    OCREngine ocr;
    TEST_CHECK(!ocr.is_initialized(), "OCREngine starts uninitialized");
    VideoEncoder venc;
    TEST_CHECK(!venc.is_initialized(), "VideoEncoder starts uninitialized");
    TEST_CHECK(venc.hidden_size() == 0, "VideoEncoder hidden 0 pre-init");
    Video vid;
    TEST_CHECK(vid.num_frames() == 0, "empty Video has 0 frames");
    TEST_CHECK(vid.fps() == 30.0f, "Video default fps 30");
    Image img;
    TEST_CHECK(img.numel() == 0, "default Image empty");
    TEST_CHECK(img.width() == 0 && img.height() == 0 && img.channels() == 0,
               "default Image dims are 0");
    AudioEncoder aenc;
    TEST_CHECK(!aenc.is_initialized(), "AudioEncoder starts uninitialized");
    TEST_CHECK(aenc.sample_rate() == 16000, "AudioEncoder default rate");
    TEST_CHECK(aenc.n_mels() == 80, "AudioEncoder default mels");
    TEST_CHECK(aenc.d_model() == 512, "AudioEncoder default d_model");
    MelSpectrogram mel;
    TEST_CHECK(!mel.is_initialized(), "MelSpectrogram starts uninitialized");
    TEST_CHECK(mel.n_mels() == 80, "MelSpectrogram default mels");
    TEST_CHECK(mel.feature_length(1640) == 8, "feature_length math (1640->8)");
    AudioFeatureExtractor afe;
    TEST_CHECK(!afe.is_initialized(), "AudioFeatureExtractor starts uninitialized");
}

static void test_sentencepiece_tsv() {
    TEST_SUITE("L072: SentencePiece TSV interchange");
    const std::string marker = "\xE2\x96\x81"; // U+2581
    {
        std::ofstream f("_test_sp.tsv");
        f << "<unk>\t-100\n";
        f << "<bos>\t-100\n";
        f << "<eos>\t-100\n";
        f << marker << "\t-2.0\n";
        f << "h\t-3.0\n";
        f << "i\t-3.0\n";
        f << (marker + "hi") << "\t-1.0\n";
    }
    UnigramTokenizer tok;
    std::string err;
    TEST_CHECK(tok.load_sentencepiece_tsv("_test_sp.tsv", &err), "TSV loads");
    TEST_CHECK(tok.vocab_size() == 7, "TSV vocab size 7");

    std::string norm = UnigramTokenizer::sentencepiece_normalize("hi  there");
    TEST_CHECK(!norm.empty() && norm.find(marker) != std::string::npos,
               "normalize inserts word marker");
    std::string joined = UnigramTokenizer::sentencepiece_join(marker + "hi" + marker + "x");
    TEST_CHECK(joined == "hi x", "join maps markers back to spaces");

    auto ids = tok.encode_sentencepiece("hi");
    TEST_CHECK(!ids.empty(), "sentencepiece encode non-empty");
    std::string back = tok.decode_sentencepiece(ids);
    TEST_CHECK(back.find("hi") != std::string::npos, "sentencepiece roundtrip keeps text");

    TEST_CHECK(tok.save_sentencepiece_tsv("_test_sp_out.tsv"), "TSV saves");
    UnigramTokenizer tok2;
    TEST_CHECK(tok2.load_sentencepiece_tsv("_test_sp_out.tsv", &err), "saved TSV reloads");
    TEST_CHECK(tok2.vocab_size() == tok.vocab_size(), "TSV save/load preserves vocab");

    TEST_CHECK(!tok.load_sentencepiece_tsv("no_such_file.tsv", &err), "missing TSV fails");
    TEST_CHECK(!err.empty(), "TSV error message set");
    std::remove("_test_sp.tsv");
    std::remove("_test_sp_out.tsv");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Transcender - Wave 7 Missing Modules (L079) + SP (L072) Suite\n");
    printf("=============================================================\n");

    test_world_model_legacy();
    test_world_model_agi();
    test_multi_agent();
    test_ocr_video_audio_contract();
    test_sentencepiece_tsv();

    printf("\n=============================================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
