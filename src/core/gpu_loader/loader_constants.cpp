#include "core/gpu_loader/loader_constants.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace layerstorm::gpu_loader {

int append_cpu_expert_device(LoaderConstants& k, int numa_node,
                             const ComputeCurve& compute) {
  const int j = k.num_devices;
  DeviceConstants dc;
  dc.position          = j;
  dc.numa_node         = numa_node;
  dc.name              = "cpu:node" + std::to_string(numa_node);
  dc.xfer_lat_us       = 0.0;      // no PCIe enqueue/detection latency
  dc.compute           = compute;  // calibrated per-expert CPU FFN curve
  dc.recon_overhead_us = 0.0;
  dc.recon_added_us    = 0.0;
  k.devices.push_back(dc);
  // No PCIe transfer from ANY bank: the CPU reads host RAM directly. rate/lat = 0
  // (tier 1 = local) so sub_[cpu] never contributes to device makespan; the read
  // pressure is priced via the source bank's egress (added per miss regardless of
  // device), not here. Every existing bank row gets one column for the new device.
  for (int b = 0; b < k.num_banks; ++b)
    k.matrix[static_cast<size_t>(b)].push_back(TransferCell{0.0, 1, 0.0});
  k.num_devices = j + 1;
  return j;
}
namespace {
using nlohmann::json;

json curve_to_json(const ComputeCurve& c) {
  return json{{"a_us", c.a_us}, {"b_us", c.b_us}, {"P", c.P}};
}
ComputeCurve curve_from_json(const json& j) {
  ComputeCurve c;
  c.a_us = j.value("a_us", 0.0);
  c.b_us = j.value("b_us", 0.0);
  c.P    = j.value("P", 1);
  return c;
}

json device_to_json(const DeviceConstants& d) {
  return json{{"position", d.position},
              {"numa_node", d.numa_node},
              {"name", d.name},
              {"uuid", d.uuid},
              {"xfer_lat_us", d.xfer_lat_us},
              {"compute", curve_to_json(d.compute)},
              {"recon_overhead_us", d.recon_overhead_us},
              {"recon_added_us", d.recon_added_us}};
}
DeviceConstants device_from_json(const json& j) {
  DeviceConstants d;
  d.position    = j.value("position", 0);
  d.numa_node   = j.value("numa_node", -1);
  d.name        = j.value("name", std::string());  // old v2 file → empty
  d.uuid        = j.value("uuid", std::string());   // old v2 file → empty (skips uuid check)
  d.xfer_lat_us = j.value("xfer_lat_us", 0.0);
  if (j.contains("compute")) d.compute = curve_from_json(j.at("compute"));
  d.recon_overhead_us = j.value("recon_overhead_us", 0.0);
  d.recon_added_us    = j.value("recon_added_us", 0.0);
  return d;
}

json bank_to_json(const BankConstants& b) {
  return json{{"node", b.node},
              {"egress_us", b.egress_us},
              {"contention", b.contention},
              {"is_hbm", b.is_hbm},
              {"cpu_affinity_node", b.cpu_affinity_node}};
}
BankConstants bank_from_json(const json& j) {
  BankConstants b;
  b.node       = j.value("node", -1);
  b.egress_us  = j.value("egress_us", 0.0);
  b.contention = j.value("contention", 1.0);
  b.is_hbm            = j.value("is_hbm", false);          // pre-HBM file → false
  b.cpu_affinity_node = j.value("cpu_affinity_node", -1);  // pre-HBM file → -1
  return b;
}

json cell_to_json(const TransferCell& c) {
  return json{{"rate_us", c.rate_us}, {"tier", c.tier}, {"lat_us", c.lat_us}};
}
TransferCell cell_from_json(const json& j) {
  TransferCell c;
  c.rate_us = j.value("rate_us", 0.0);
  c.tier    = j.value("tier", 1);
  c.lat_us  = j.value("lat_us", 0.0);
  return c;
}

json cont_to_json(const ContentionSample& s) {
  return json{{"bank", s.bank},
              {"device_a", s.device_a},
              {"device_b", s.device_b},
              {"config", static_cast<int>(s.config)},
              {"solo_us", s.solo_us},
              {"dual_us", s.dual_us},
              {"factor", s.factor}};
}
ContentionSample cont_from_json(const json& j) {
  ContentionSample s;
  s.bank     = j.value("bank", -1);
  s.device_a = j.value("device_a", -1);
  s.device_b = j.value("device_b", -1);
  s.config   = static_cast<ContentionConfig>(j.value("config", 0));
  s.solo_us  = j.value("solo_us", 0.0);
  s.dual_us  = j.value("dual_us", 0.0);
  s.factor   = j.value("factor", 1.0);
  return s;
}

json build(const LoaderConstants& c) {
  json j;
  j["version"]      = c.version;
  j["source"]       = c.source;
  j["expert_bytes"] = c.expert_bytes;
  j["num_devices"]  = c.num_devices;
  j["num_banks"]    = c.num_banks;
  j["compute_N"]            = c.compute_N;
  j["compute_K"]            = c.compute_K;
  j["compute_tokens"]       = c.compute_tokens;
  j["recon_payload_bytes"]  = c.recon_payload_bytes;
  j["fixed_overhead_us"]    = c.fixed_overhead_us;
  // Trained place_cons weight (only w_numa; residency+hotness are heuristics): emit
  // only when present (old files stay clean).
  if (c.place_sum_weights.present) {
    j["place_sum_weights"] = {
        {"w_numa", c.place_sum_weights.w_numa},
    };
  }
  j["ncf"]          = c.ncf;

  json devs = json::array();
  for (const auto& d : c.devices) devs.push_back(device_to_json(d));
  j["devices"] = std::move(devs);

  json banks = json::array();
  for (const auto& b : c.banks) banks.push_back(bank_to_json(b));
  j["banks"] = std::move(banks);

  json mat = json::array();
  for (const auto& row : c.matrix) {
    json jrow = json::array();
    for (const auto& cell : row) jrow.push_back(cell_to_json(cell));
    mat.push_back(std::move(jrow));
  }
  j["matrix"] = std::move(mat);

  json cont = json::array();
  for (const auto& s : c.contention) cont.push_back(cont_to_json(s));
  j["contention"] = std::move(cont);
  return j;
}

LoaderConstants parse(const json& j) {
  LoaderConstants c;
  c.version      = j.value("version", kLoaderConstantsVersion);
  c.source       = j.value("source", std::string("loaded"));
  c.expert_bytes = j.value("expert_bytes", 0.0);
  c.num_devices  = j.value("num_devices", 0);
  c.num_banks    = j.value("num_banks", 0);
  c.compute_N           = j.value("compute_N", 0);
  c.compute_K           = j.value("compute_K", 0);
  c.compute_tokens      = j.value("compute_tokens", 0);
  c.recon_payload_bytes = j.value("recon_payload_bytes", 0.0);
  c.fixed_overhead_us   = j.value("fixed_overhead_us", 0.0);  // old files → 0 (no overhead)
  if (j.contains("place_sum_weights")) {
    const auto& pw = j.at("place_sum_weights");
    c.place_sum_weights.present = true;
    c.place_sum_weights.w_numa  = pw.value("w_numa", 0.0);
  }
  c.ncf          = j.value("ncf", std::vector<double>{});
  if (j.contains("devices"))
    for (const auto& d : j.at("devices")) c.devices.push_back(device_from_json(d));
  if (j.contains("banks"))
    for (const auto& b : j.at("banks")) c.banks.push_back(bank_from_json(b));
  if (j.contains("matrix")) {
    for (const auto& jrow : j.at("matrix")) {
      std::vector<TransferCell> row;
      for (const auto& cell : jrow) row.push_back(cell_from_json(cell));
      c.matrix.push_back(std::move(row));
    }
  }
  if (j.contains("contention"))
    for (const auto& s : j.at("contention")) c.contention.push_back(cont_from_json(s));
  return c;
}

}  // namespace

std::string to_json_string(const LoaderConstants& c, int indent) {
  return build(c).dump(indent);
}

LoaderConstants from_json_string(const std::string& s) {
  return parse(json::parse(s));
}

void save(const LoaderConstants& c, const std::string& path) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("gpu_loader: cannot open for write: " + path);
  f << build(c).dump(2) << '\n';
}

LoaderConstants load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("gpu_loader: cannot open for read: " + path);
  return parse(json::parse(f));
}

bool compute_dims_match(const LoaderConstants& c, int model_N, int model_K) {
  if (c.compute_N == 0) return true;  // transfer-only file: nothing model-specific to mistrust
  return c.compute_N == model_N && c.compute_K == model_K;
}

}  // namespace layerstorm::gpu_loader

namespace layerstorm::gpu_loader {

M2Params load_m2_params(const std::string& path, int num_devices) {
  M2Params p;
  std::ifstream in(path);
  if (!in.good()) return p;
  nlohmann::json j;
  try {
    in >> j;
    if (j.value("form", "") != "v2") return p;
    auto vec = [&](const char* key, std::vector<double>& out) {
      out = j.at(key).get<std::vector<double>>();
      return static_cast<int>(out.size()) == num_devices;
    };
    if (!vec("s", p.s) || !vec("o0", p.o0) || !vec("oc", p.oc) ||
        !vec("hsat", p.hsat) || !vec("cpw", p.cpw))
      return p;
    p.g_c = j.value("g_c", 1.0);
    for (double& h : p.hsat)
      if (h < 1e-6) h = 1e-6;
    p.valid = true;
  } catch (const std::exception&) {
    p = M2Params{};
  }
  return p;
}

}  // namespace layerstorm::gpu_loader
