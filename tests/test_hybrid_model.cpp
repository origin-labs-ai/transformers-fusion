#include "quant/hybrid_block.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("hybrid_model");
 printf("=== HybridModel wiring tests ===\n\n");
 printf("--- T1: 93-layer K3 schedule wiring ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=32; cfg.num_heads=2; cfg.head_dim=16; cfg.num_layers=93; cfg.max_seq_len=32; cfg.ffn_hidden_size=64;
  cfg.q_lora_rank=8; cfg.kv_lora_rank=8; cfg.mla_rope_dim=4;
  auto sched=build_k3_schedule();
  // adjust to 93 layers explicitly for test (build_k3_schedule is 93)
  TEST_CHECK(sched.total()==93,"schedule 93");
  HybridModel model(cfg,sched);
  int64_t pc=model.param_count();
  printf("  param_count=%lld KDA=%d MLA=%d\n",(long long)pc,sched.num_kda,sched.num_mla);
  TEST_CHECK(pc>0,"params >0");
  TEST_CHECK(sched.num_kda>=68 && sched.num_kda<=71,"KDA count");
 }
 printf("--- T2: forward shape/finite ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=16; cfg.num_heads=2; cfg.head_dim=8; cfg.num_layers=4; cfg.max_seq_len=16; cfg.ffn_hidden_size=32; cfg.q_lora_rank=4; cfg.kv_lora_rank=4; cfg.mla_rope_dim=4;
  auto sched=build_hybrid_schedule(4,3);
  HybridModel model(cfg,sched);
  Tensor x(Shape{1,4,16}); for(int64_t i=0;i<x.numel();++i) x.data<float>()[i]=(float)(i%5)*0.1f;
  Tensor pos(Shape{1,4}); for(int i=0;i<4;++i) pos.data<float>()[i]=(float)i;
  Tensor y=model.forward(x,pos,nullptr);
  TEST_CHECK(y.dim(0)==1 && y.dim(1)==4 && y.dim(2)==16,"shape");
  bool fin=true; for(int64_t i=0;i<y.numel();++i) if(!std::isfinite(y.data<float>()[i])) fin=false;
  TEST_CHECK(fin,"finite");
  printf("  y[0]=%.4f\n",y.data<float>()[0]);
 }
 printf("\nHybrid wiring TESTS PASSED!\n"); return 0;
}
