#include "quant/thread_safety.h"

namespace quant {

// Explicit template instantiations for common types
template class SPSCQueue<int>;
template class SPSCQueue<float>;
template class SPSCQueue<int64_t>;

} // namespace quant
