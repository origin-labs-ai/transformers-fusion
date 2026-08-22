#include "quant/rollout_worker.h"
#include "quant/model.h"
#include "quant/moe_model.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("rollout");
 printf("=== RolloutWorker (G-5) tests ===\n\n");
 printf("--- T1: dense rollout ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=32; cfg.num_heads=2; cfg.head_dim=16; cfg.num_layers=1; cfg.vocab_size=64; cfg.max_seq_len=32; cfg.ffn_hidden_size=64;
  DenseModel model(cfg);
  RolloutWorker w(&model,nullptr,RolloutConfig{2,8,0.8f,32});
  auto toks=w.generate("hello");
  printf("  toks=%zu\n",toks.size());
  TEST_CHECK(toks.size()>=5 && toks.size()<= 13,"length in range");
  bool finite=true; for(auto t:toks) if(t<0||t>=64) finite=false;
  TEST_CHECK(finite,"tokens in vocab");
 }
 printf("--- T2: batch ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=32; cfg.num_heads=2; cfg.head_dim=16; cfg.num_layers=1; cfg.vocab_size=64; cfg.max_seq_len=32; cfg.ffn_hidden_size=64;
  DenseModel model(cfg);
  RolloutWorker w(&model,nullptr);
  auto batch=w.generate_batch({"hi","hello world"});
  TEST_CHECK(batch.size()==2,"batch size 2");
 }
 printf("\nRollout TESTS PASSED!\n"); return 0;
}
