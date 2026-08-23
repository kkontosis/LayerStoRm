// AttentionDevice factory: dispatches AttentionBackendType to concrete
// AttentionDevice implementations.

#include "core/attention_device.h"
#include "compute/snapmla_sm120_attention_device.h"
#include "compute/csa_hca_sm120_attention_device.h"
#include "compute/tq_sm120_attention_device.h"
#include "config/config_parser.h"

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

std::unique_ptr<AttentionDevice> make_attention_device(
        config::AttentionBackendType type, config::GpuRef gpu) {
    switch (type) {
    case config::AttentionBackendType::snapmla:
        return make_snapmla_sm120_attention_device(std::move(gpu));
    case config::AttentionBackendType::turboquant_mla:
        return make_tq_sm120_attention_device(std::move(gpu));
    case config::AttentionBackendType::csa_hca:
    case config::AttentionBackendType::csa_hca_tq:
    case config::AttentionBackendType::csa_hca_tq_mix:
        // V4-5T (TD-V4-TQ-DEVICE resolved): ONE V4 arch device for every
        // csa_hca* arm — the per-tier CODEC (fp8 vs tq4) is baked at
        // engine configure time from the allocator's per-tier formats
        // (V4DeviceOptions::csa_codec/hca_codec + csa_hca_device_set_tq);
        // the arch × codec composition of the attention refactor V2
        // (daemon/attention/kv_codec.h documents the map).
        return make_csa_hca_sm120_attention_device(std::move(gpu));
    default:
        throw std::runtime_error(
            "Unknown AttentionBackendType: " +
            std::to_string(static_cast<int>(type)));
    }
}

}  // namespace layerstorm::compute
