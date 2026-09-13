#include "quant/hybrid_scheduler.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("hybrid");
 printf("=== Hybrid Scheduler all-STD (owner purge 2026-09-07) tests ===\n\n");
 printf("--- T1: 93 layers all STD ---\n");
 {
  auto s=build_hybrid_schedule(93,0);
  printf("  STD=%d total=%d\n",s.num_std,s.total());
  TEST_CHECK(s.total()==93,"93 layers");
  TEST_CHECK(s.num_std==93,"93 STD");
  bool allstd=true;
  for(auto k:s.layers) if(k!=HybridAttnKind::STD) allstd=false;
  TEST_CHECK(allstd,"all STD");
 }
 printf("--- T2: generic all-STD ---\n");
 {
  for(int n:{12,24,48,96}){
   auto s=build_hybrid_schedule(n,1);
   printf("  n=%d STD=%d\n",n,s.num_std);
   TEST_CHECK(s.num_std==n,"STD == n");
   TEST_CHECK(s.layers[(size_t)n-1]==HybridAttnKind::STD,"tail STD");
  }
 }
 printf("--- T3: determinism ---\n");
 {
  auto a=build_hybrid_schedule(93,3);
  auto b=build_hybrid_schedule(93,3);
  int same=1;
  for(int i=0;i<93;++i) if(a.layers[(size_t)i]!=b.layers[(size_t)i]) same=0;
  TEST_CHECK(same,"deterministic");
 }
 printf("\nHybrid TESTS DONE.\n");
 int fails = TEST_REPORT();
 return fails > 0 ? 1 : 0;
}
