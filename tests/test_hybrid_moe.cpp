#include "quant/hybrid_moe_model.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("hybrid_moe");
 printf("=== HybridMoE wiring tests ===\n\n");
 printf("--- T1: 4-layer hybrid MoE ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=16; cfg.num_heads=2; cfg.head_dim=8; cfg.num_layers=4; cfg.max_seq_len=16; cfg.ffn_hidden_size=32; cfg.vocab_size=64; cfg.q_lora_rank=4; cfg.kv_lora_rank=4; cfg.mla_rope_dim=4;
  moe::MoEAllConfig mcfg; mcfg.num_experts=4; mcfg.top_k=2; mcfg.expert_hidden_size=32;
  auto sched=build_hybrid_schedule(4,3);
  HybridMoeModel model(cfg,mcfg,sched);
  TEST_CHECK(model.param_count()>0,"params >0");
  printf("  params=%lld\n",(long long)model.param_count());
 }
 printf("--- T2: forward shape/finite ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=16; cfg.num_heads=2; cfg.head_dim=8; cfg.num_layers=2; cfg.max_seq_len=16; cfg.ffn_hidden_size=32; cfg.vocab_size=32; cfg.q_lora_rank=4; cfg.kv_lora_rank=4; cfg.mla_rope_dim=4;
  moe::MoEAllConfig mcfg; mcfg.num_experts=2; mcfg.top_k=1; mcfg.expert_hidden_size=32;
  auto sched=build_hybrid_schedule(2,3);
  HybridMoeModel model(cfg,mcfg,sched);
  Tensor ids(Shape{1,4}); Tensor pos(Shape{1,4}); for(int i=0;i<4;++i){ ids.data<float>()[i]=(float)(i%30+1); pos.data<float>()[i]=(float)i; }
  Tensor y=model.forward(ids,pos,nullptr);
  TEST_CHECK(y.dim(0)==1 && y.dim(1)==4,"shape");
  bool fin=true; for(int64_t i=0;i<y.numel();++i) if(!std::isfinite(y.data<float>()[i])) fin=false;
  TEST_CHECK(fin,"finite");
  printf("  y[0]=%.4f\n",y.data<float>()[0]);
 }
 printf("\nHybridMoE TESTS PASSED!\n"); return 0;
}
