#include "quant/hybrid_scheduler.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("hybrid");
 printf("=== Hybrid Scheduler (G-3) tests ===\n\n");
 printf("--- T1: K3 93 layers 3:1 ---\n");
 {
  auto s=build_k3_schedule();
  printf("  KDA=%d MLA=%d total=%d\n",s.num_kda,s.num_mla,s.total());
  TEST_CHECK(s.total()==93,"93 layers");
  TEST_CHECK(s.num_kda>=68 && s.num_kda<=70,"KDA ~69");
  TEST_CHECK(s.num_mla>=23 && s.num_mla<=25,"MLA ~24");
  int mla_spacing_ok=1;
  for(int i=0;i<(int)s.layers.size();++i) if(s.layers[i]==HybridAttnKind::MLA){
    // MLA not adjacent
    if(i+1<(int)s.layers.size() && s.layers[i+1]==HybridAttnKind::MLA) mla_spacing_ok=0;
  }
  TEST_CHECK(mla_spacing_ok,"MLA not adjacent (interleaved)");
 }
 printf("--- T2: ratio holds ---\n");
 {
  for(int n:{12,24,48,96}){
   auto s=build_hybrid_schedule(n,3);
   double ratio = s.num_mla? (double)s.num_kda/s.num_mla : 0;
   printf("  n=%d KDA=%d MLA=%d ratio=%.2f\n",n,s.num_kda,s.num_mla,ratio);
   TEST_CHECK(ratio>2.5 && ratio<3.5,"ratio ~3:1");
  }
 }
 printf("--- T3: determinism ---\n");
 {
  auto a=build_hybrid_schedule(93,3);
  auto b=build_hybrid_schedule(93,3);
  int same=1;
  for(int i=0;i<93;++i) if(a.layers[i]!=b.layers[i]) same=0;
  TEST_CHECK(same,"deterministic");
 }
 printf("\nHybrid TESTS PASSED!\n"); return 0;
}
