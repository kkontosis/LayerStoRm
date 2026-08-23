#pragma once

#include "config/config_parser.h"
#include "daemon/ipc_protocol.h"

namespace layerstorm::daemon {

/// Apply config update entries from a CMD_CONFIG_UPDATE command to the
/// daemon's local config copy. Returns the number of fields successfully
/// updated. Rejects non-changeable fields (skips silently).
uint32_t apply_config_update(config::Config& cfg,
                             const ipc::Command& cmd);

}  // namespace layerstorm::daemon
