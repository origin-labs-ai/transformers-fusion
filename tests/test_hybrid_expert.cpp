#include "quant/hybrid_expert.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("hybrid_expert");
 printf("=== HybridExpertShard tests ===\n\n");
 expert::ClusterConfig cfg; cfg.num_experts=64;
 printf("--- T1: single layer round-robin ---\n");
 {
  HybridExpertShard sh(cfg, 4);
  auto a=sh.assign_layer(8);
  TEST_CHECK(a.size()==8,"8 experts");
  // round-robin: expert 0->node0, 1->1, 4->0 etc
  TEST_CHECK(a[0].node_id==0 && a[1].node_id==1 && a[4].node_id==0,"round-robin");
 }
 printf("--- T2: schedule 93 layers ---\n");
 {
  HybridExpertShard sh(cfg, 8);
  auto sched=build_hybrid_schedule(93,0);
  auto per=sh.assign_schedule(sched);
  TEST_CHECK((int)per.size()==93,"93 layers assignments");
  // each layer should have 64 entries
  TEST_CHECK(per[0].size()==64,"64 per layer");
  // balanced: each node gets 8 experts per layer (64/8)
  int cnt[8]={0};
  for(auto &a: per[0]) cnt[a.node_id]++;
  for(int i=0;i<8;++i) TEST_CHECK(cnt[i]==8,"balanced 8 per node");
 }
 printf("\nHybridExpert TESTS DONE.\n");
 int fails = TEST_REPORT();
 return fails > 0 ? 1 : 0;
}
