#include "quant/sops_integration.h"

namespace quant {

SopsGlobalState& sops_global() {
    static SopsGlobalState state;
    return state;
}

} // namespace quant
