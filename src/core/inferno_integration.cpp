#include "quant/inferno_integration.h"

namespace quant {

InfernoGlobalState& inferno_global() {
    static InfernoGlobalState state;
    return state;
}

} // namespace quant
