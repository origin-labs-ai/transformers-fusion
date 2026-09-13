#include "quant/hybrid_block.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("hybrid_model");
 printf("=== HybridModel wiring tests (all-STD, owner purge 2026-09-07) ===\n\n");
 printf("--- T1: 93-layer schedule wiring (all STD) ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=32; cfg.num_heads=2; cfg.head_dim=16; cfg.num_layers=93; cfg.max_seq_len=32; cfg.ffn_hidden_size=64;
  auto sched=build_hybrid_schedule(93,0);
  TEST_CHECK(sched.total()==93,"schedule 93");
  TEST_CHECK(sched.num_std==93,"93 STD");
  HybridModel model(cfg,sched);
  int64_t pc=model.param_count();
  printf("  param_count=%lld STD=%d\n",(long long)pc,sched.num_std);
  TEST_CHECK(pc>0,"params >0");
 }
 printf("--- T2: stack forward shape/finite (STD dispatch) ---\n");
 {
  TransformerConfig cfg; cfg.hidden_size=16; cfg.num_heads=2; cfg.head_dim=8; cfg.num_layers=4; cfg.max_seq_len=16; cfg.ffn_hidden_size=32;
  auto sched=build_hybrid_schedule(4,1);
  TEST_CHECK(sched.num_std==4,"4 STD");
  TEST_CHECK(sched.layers[3]==HybridAttnKind::STD,"tail STD");
  HybridModel model(cfg,sched);
  Tensor x(Shape{1,4,16}); for(int64_t i=0;i<x.numel();++i) x.data<float>()[i]=(float)(i%5)*0.1f;
  Tensor pos(Shape{1,4}); for(int i=0;i<4;++i) pos.data<float>()[i]=(float)i;
  Tensor y=model.forward(x,pos,nullptr);
  TEST_CHECK(y.dim(0)==1 && y.dim(1)==4 && y.dim(2)==16,"shape");
  bool fin=true; for(int64_t i=0;i<y.numel();++i) if(!std::isfinite(y.data<float>()[i])) fin=false;
  TEST_CHECK(fin,"finite");
  printf("  y[0]=%.4f\n",y.data<float>()[0]);
 }
 printf("\nHybrid wiring TESTS DONE.\n");
 int fails = TEST_REPORT();
 return fails > 0 ? 1 : 0;
}
