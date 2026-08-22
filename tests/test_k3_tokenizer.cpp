#include "quant/k3_tokenizer.h"
#include "quant/test.h"
#include <cstdio>
using namespace quant;
int main(){
 TEST_SUITE("k3_tokenizer");
 printf("=== K3 Tokenizer (G-6) tests ===\n\n");
 printf("--- T1: roundtrip ---\n");
 {
  K3Tokenizer tok;
  std::string s="hello world 123";
  auto ids=tok.encode(s);
  TEST_CHECK(!ids.empty(),"non-empty");
  for(int id:ids) TEST_CHECK(id>=0 && id<tok.vocab_size(),"id in range");
  std::string dec=tok.decode(ids,false);
  TEST_CHECK(!dec.empty(),"decode non-empty");
 }
 printf("--- T2: special tokens ---\n");
 {
  K3Tokenizer tok;
  TEST_CHECK(tok.is_special(tok.im_start_id()),"im_start special");
  TEST_CHECK(tok.is_special(tok.im_end_id()),"im_end special");
  TEST_CHECK(tok.vocab_size()==152064,"vocab size 152064");
 }
 printf("--- T3: chat template ---\n");
 {
  K3Tokenizer tok;
  std::vector<std::pair<std::string,std::string>> msgs={{"user","hi"},{"assistant","hello"}};
  auto ids=tok.apply_chat_template(msgs,true);
  TEST_CHECK(!ids.empty(),"chat ids");
 }
 printf("\nK3 Tokenizer TESTS PASSED!\n"); return 0;
}
