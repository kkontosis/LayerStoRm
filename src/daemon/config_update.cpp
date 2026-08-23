#include "config_update.h"

#include <spdlog/spdlog.h>

namespace layerstorm::daemon {

uint32_t apply_config_update(config::Config& cfg,
                             const ipc::Command& cmd) {
    const auto& cu = cmd.config_update;
    const uint32_t count = cu.count;
    if (count > 29) {
        spdlog::warn("config_update: count {} exceeds max 29", count);
        return 0;
    }

    uint32_t applied = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& entry = cu.entries[i];
        auto fid = static_cast<config::FieldId>(entry.field_id);

        if (!config::is_changeable(fid)) {
            spdlog::warn("config_update: field {} ({}) is not changeable",
                         entry.field_id, config::field_name(fid));
            continue;
        }

        if (config::apply_field_update(cfg, fid, entry.value_type, entry.raw_value)) {
            spdlog::debug("config_update: applied {} = raw({})",
                          config::field_name(fid), entry.raw_value);
            ++applied;
        } else {
            spdlog::warn("config_update: type mismatch for {} (value_type={})",
                         config::field_name(fid), entry.value_type);
        }
    }
    return applied;
}

}  // namespace layerstorm::daemon
