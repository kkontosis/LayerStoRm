// SpeculationMethod factory implementation.  See speculation_factory.h for
// the pattern rationale (explicit switch, mirrors attention_device_factory;
// registry rejected — static-lib initializer dropping).

#include "speculation/speculation_factory.h"

#include "speculation/dspark_speculation_method.h"
#include "speculation/mtp_speculation_method.h"
#include "speculation/null_speculation_method.h"
#include "config/config_parser.h"

#include <stdexcept>
#include <string>

namespace layerstorm::speculation {

// NullSpeculationMethod::type() lives here (not the header) so the
// header stays free of config_parser.h (mirrors how attention_device.h
// forward-declares AttentionBackendType).
config::SpeculationMethodType NullSpeculationMethod::type() const {
    return config::SpeculationMethodType::null;
}

const char* speculation_method_name(config::SpeculationMethodType type) {
    switch (type) {
    case config::SpeculationMethodType::none:          return "none";
    case config::SpeculationMethodType::null:          return "null";
    case config::SpeculationMethodType::mtp:           return "mtp";
    case config::SpeculationMethodType::prompt_lookup: return "prompt_lookup";
    case config::SpeculationMethodType::dspark:        return "dspark";
    }
    return "unknown";
}

bool has_dspark(const config::Config& cfg) {
    return cfg.speculation.method == config::SpeculationMethodType::dspark;
}

std::unique_ptr<SpeculationMethod> make_speculation_method(
        config::SpeculationMethodType type, const SpeculationInitContext& ctx) {
    switch (type) {
    case config::SpeculationMethodType::none:
        return nullptr;  // subsystem off (DEFAULT) — nothing constructed

    case config::SpeculationMethodType::null: {
        auto method = std::make_unique<NullSpeculationMethod>();
        method->init(ctx);
        return method;
    }

    // #16 / GLM-25g: MTP self-draft off the frozen trunk.  init() fails
    // closed on configs/checkpoints without an MTP head (see
    // MtpSpeculationMethod::init).
    case config::SpeculationMethodType::mtp: {
        auto method = std::make_unique<MtpSpeculationMethod>();
        method->init(ctx);
        return method;
    }

    // Reserved selector: fail closed at engine init instead of silently
    // running without the requested speculation.
    case config::SpeculationMethodType::prompt_lookup:
        throw std::runtime_error(
            "speculation.method=prompt_lookup is reserved: the prompt-lookup "
            "SpeculationMethod is not implemented yet");

    // DSP-5: the DSpark drafter.  The method owns policy + the greedy
    // acceptance rule; the draft EXECUTION stays engine machinery — the
    // DsparkRuntime armed by engine.cpp step 19e (draft-checkpoint upload +
    // aux-hidden export + D_CMD_RUN_DSPARK_STEP), reached through the
    // driver-installed DsparkStepExecutor.  init() fails closed on a
    // self-inconsistent speculation.dspark; the checkpoint / draft GPU
    // fail-closed surface remains DsparkRuntime::create.
    case config::SpeculationMethodType::dspark: {
        auto method = std::make_unique<DsparkSpeculationMethod>();
        method->init(ctx);
        return method;
    }
    }

    throw std::runtime_error(
        "Unknown SpeculationMethodType: " +
        std::to_string(static_cast<int>(type)));
}

std::unique_ptr<SpeculationMethod> make_null_speculation_method() {
    auto method = std::make_unique<NullSpeculationMethod>();
    method->init(SpeculationInitContext{});
    return method;
}

}  // namespace layerstorm::speculation
