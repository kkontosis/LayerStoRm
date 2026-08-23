#include "core/gpu_loader/loader_policy.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace layerstorm::gpu_loader {
namespace {
using nlohmann::json;

json build(const PolicyParams& p) {
  return json{{"epoch", p.epoch},
              {"freq_w", p.freq_w},
              {"freq_decay", p.freq_decay},
              {"reuse_w", p.reuse_w},
              {"reuse_tau", p.reuse_tau}};
}

// Fail-loud field read: absent -> default; present-but-not-a-number -> throw.
double num_or(const json& j, const char* key, double def) {
  if (!j.contains(key)) return def;
  const json& v = j.at(key);
  if (!v.is_number())
    throw std::runtime_error(std::string("loader_policy: field '") + key +
                             "' must be a number");
  return v.get<double>();
}

PolicyParams parse(const json& j) {
  if (!j.is_object())
    throw std::runtime_error("loader_policy: artifact root must be a JSON object");
  PolicyParams p;
  if (j.contains("epoch")) {
    if (!j.at("epoch").is_string())
      throw std::runtime_error("loader_policy: field 'epoch' must be a string");
    p.epoch = j.at("epoch").get<std::string>();
  }
  p.freq_w     = num_or(j, "freq_w", p.freq_w);
  p.freq_decay = num_or(j, "freq_decay", p.freq_decay);
  p.reuse_w    = num_or(j, "reuse_w", p.reuse_w);
  p.reuse_tau  = num_or(j, "reuse_tau", p.reuse_tau);
  // Range validation: a fitted artifact carrying out-of-domain values is a bad
  // fit — fail loud, never silently clamp (env overrides clamp; artifacts don't).
  if (p.freq_w < 0.0)
    throw std::runtime_error("loader_policy: freq_w must be >= 0");
  if (p.freq_decay <= 0.0)
    throw std::runtime_error("loader_policy: freq_decay must be > 0");
  if (p.reuse_w < 0.0)
    throw std::runtime_error("loader_policy: reuse_w must be >= 0");
  if (p.reuse_tau <= 0.0)
    throw std::runtime_error("loader_policy: reuse_tau must be > 0");
  return p;
}

}  // namespace

std::string to_json_string(const PolicyParams& p, int indent) {
  return build(p).dump(indent);
}

PolicyParams policy_params_from_json_string(const std::string& s) {
  json j;
  try {
    j = json::parse(s);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("loader_policy: JSON parse failed: ") +
                             e.what());
  }
  return parse(j);
}

PolicyParams load_policy_params(const std::string& path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("loader_policy: cannot open for read: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  try {
    return policy_params_from_json_string(ss.str());
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(e.what()) + " (" + path + ")");
  }
}

bool apply_policy_env_overrides(PolicyParams& p) {
  bool any = false;
  if (const char* fw = std::getenv("LS_LOADER_FREQ_W"); fw && *fw) {
    p.freq_w = std::atof(fw);
    if (p.freq_w < 0.0) p.freq_w = 0.0;
    any = true;
  }
  if (const char* fd = std::getenv("LS_LOADER_FREQ_DECAY"); fd && *fd) {
    p.freq_decay = std::atof(fd);
    if (p.freq_decay <= 0.0) p.freq_decay = 0.1;
    any = true;
  }
  if (const char* rw = std::getenv("LS_LOADER_REUSE_W"); rw && *rw) {
    p.reuse_w = std::atof(rw);
    if (p.reuse_w < 0.0) p.reuse_w = 0.0;
    any = true;
  }
  if (const char* rt = std::getenv("LS_LOADER_REUSE_TAU"); rt && *rt) {
    p.reuse_tau = std::atof(rt);
    if (p.reuse_tau <= 0.0) p.reuse_tau = 300.0;
    any = true;
  }
  return any;
}

}  // namespace layerstorm::gpu_loader
