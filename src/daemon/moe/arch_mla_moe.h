// ArchMlaMoe — GLM / DeepSeek-V3.2 (MLA) family MoE arch participant.
//
// The family's MoE behavior IS the common base path: learned top-K
// self-gating (V4-4a scoring_func parametrizes sigmoid), plain residual
// add, quantized shared-expert route, dense first-k layers via the
// config-driven driver early-out. It therefore overrides NOTHING; the
// MoeArch base default hook bodies (the family's named home) are defined
// in arch_mla_moe.cpp. See arch_base.h for the INV-MOE-ARCH contract.

#pragma once

#include "daemon/moe/arch_base.h"

namespace layerstorm::daemon {

class ArchMlaMoe final : public MoeArch {
public:
    explicit ArchMlaMoe(CommandDispatcher& d) : MoeArch(d) {}
};

}  // namespace layerstorm::daemon
