"""Autoconfig foundation tests: detection, model shape, sizing transcription,
calibration accept, fingerprint, GGUF metadata (TD-AUTOCONFIG-HARDWARE-FIT).

All CPU-only and hermetic: hardware comes from a fake /proc + /sys tree that
replicates the dev box (2x5090 + 2x5080, 4 DDR nodes + 4 CPU-less HBM banks),
calibration from a synthetic artifact mirroring
test-data/gpu_loader_calibration_ep4x4.json.
"""

import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from autoconfig import gguf_meta, sizing
from autoconfig.calibration import (CalibrationView, accept_calibration,
                                    load_calibration, resolve_calibration_path)
from autoconfig.engine_constraints import ENGINE_CONSTRAINTS, active_for, get
from autoconfig.fingerprint import fingerprint
from autoconfig.hwdetect import HardwareDescriptor, detect_hardware
from autoconfig.modelshape import LinearAttnGeometry, ModelShape

REPO = os.path.join(os.path.dirname(__file__), "..", "..")

# Dev-box GPU facts (procfs + sysfs ground truth captured 2026-08-30).
BOX_GPUS = [
    # (bus_id, model, uuid, bar1_gib, numa)
    ("0000:16:00.0", "NVIDIA GeForce RTX 5080", "GPU-845beaa5-2b9d-8d88-ed40-7c08ce810bb2", 16, 0),
    ("0000:40:00.0", "NVIDIA GeForce RTX 5080", "GPU-08f6631f-daee-e9af-af69-d03fb74d184a", 16, 2),
    ("0000:6a:00.0", "NVIDIA GeForce RTX 5090", "GPU-0bb7512a-f3dc-fb8d-ad30-84fc27f76f3b", 32, 2),
    ("0000:94:00.0", "NVIDIA GeForce RTX 5090", "GPU-11ce54c2-ef9f-197f-a5da-179a27928d43", 32, 3),
]


def build_fake_tree(root, gpus=BOX_GPUS, ddr_nodes=4, hbm_nodes=4,
                    mem_total_kib=595088256, pcie_max="32.0 GT/s PCIe"):
    proc = os.path.join(root, "proc")
    sysd = os.path.join(root, "sys")
    for bus, model, uuid, bar1_gib, numa in gpus:
        d = os.path.join(proc, "driver/nvidia/gpus", bus)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "information"), "w") as f:
            f.write(f"Model: \t\t {model}\nIRQ:   \t\t 16\nGPU UUID: \t {uuid}\n")
        p = os.path.join(sysd, "bus/pci/devices", bus)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "numa_node"), "w") as f:
            f.write(f"{numa}\n")
        with open(os.path.join(p, "max_link_speed"), "w") as f:
            f.write(pcie_max + "\n")
        with open(os.path.join(p, "max_link_width"), "w") as f:
            f.write("16\n")
        with open(os.path.join(p, "current_link_speed"), "w") as f:
            f.write("2.5 GT/s PCIe\n")  # the idle-downclock trap
        bar1 = bar1_gib << 30
        with open(os.path.join(p, "resource"), "w") as f:
            f.write("0x00000000a0000000 0x00000000a0ffffff 0x0000000000040200\n")
            f.write(f"0x0000004000000000 0x{0x4000000000 + bar1 - 1:016x} 0x000000000014220c\n")
            f.write("0x0000000000000000 0x0000000000000000 0x0000000000000000\n")
    for n in range(ddr_nodes + hbm_nodes):
        d = os.path.join(sysd, "devices/system/node", f"node{n}")
        os.makedirs(d, exist_ok=True)
        is_hbm = n >= ddr_nodes
        with open(os.path.join(d, "cpulist"), "w") as f:
            f.write("" if is_hbm else f"{n*14}-{n*14+13}\n")
        total = 16777216 if is_hbm else 131699660
        with open(os.path.join(d, "meminfo"), "w") as f:
            f.write(f"Node {n} MemTotal:       {total} kB\n")
            f.write(f"Node {n} MemFree:        {total // 2} kB\n")
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(proc, "meminfo"), "w") as f:
        f.write(f"MemTotal:       {mem_total_kib} kB\nMemFree:        100000000 kB\n")
    nv = os.path.join(sysd, "class/nvme/nvme0")
    dev = os.path.join(nv, "device")
    os.makedirs(dev, exist_ok=True)
    with open(os.path.join(nv, "model"), "w") as f:
        f.write("Samsung SSD 990 PRO 4TB\n")
    with open(os.path.join(dev, "current_link_speed"), "w") as f:
        f.write("8.0 GT/s PCIe\n")
    with open(os.path.join(dev, "current_link_width"), "w") as f:
        f.write("4\n")
    return proc, sysd


@pytest.fixture()
def box_hw(tmp_path):
    proc, sysd = build_fake_tree(str(tmp_path))
    return detect_hardware(proc_root=proc, sys_root=sysd)


def glm52_shape():
    with open(os.path.join(REPO, "recipes", "glm52_serve_champion.json")) as f:
        return ModelShape.from_model_section(json.load(f)["model"])


# ------------------------------------------------------------- detection

class TestHwDetect:
    def test_gpu_enumeration_pci_order(self, box_hw):
        assert [g.pci_bus_id for g in box_hw.gpus] == [b for b, *_ in BOX_GPUS]
        assert [g.gpu_type for g in box_hw.gpus] == ["rtx5080", "rtx5080", "rtx5090", "rtx5090"]
        assert [g.uuid for g in box_hw.gpus] == [u for _, _, u, _, _ in BOX_GPUS]

    def test_vram_from_bar1_not_name(self, box_hw):
        assert [g.vram_mib for g in box_hw.gpus] == [16384, 16384, 32768, 32768]
        assert all(g.vram_source == "bar1" for g in box_hw.gpus)

    def test_pcie_gen_uses_max_not_current(self, box_hw):
        # current_link_speed says 2.5 GT/s (idle downclock); max is the truth
        assert all(g.pcie_gen_max == 5 for g in box_hw.gpus)
        assert all(g.pcie_width_max == 16 for g in box_hw.gpus)

    def test_numa_hbm_banks(self, box_hw):
        assert box_hw.hbm_nodes() == (4, 5, 6, 7)
        assert box_hw.gpu_attached_nodes() == (0, 2, 3)
        hbm = [n for n in box_hw.numa_nodes if n.is_hbm_bank]
        assert all(abs(n.mem_total_kib - 16777216) < 1024 for n in hbm)

    def test_host_ram(self, box_hw):
        assert abs(box_hw.mem_total_gib - 567.5) < 1.0

    def test_nvme_gen3_x4_ceiling(self, box_hw):
        nv = box_hw.nvmes[0]
        assert (nv.pcie_gen, nv.pcie_width) == (3, 4)
        assert 3.0 < nv.bandwidth_ceiling_gbps < 3.6  # the ~3.3 GB/s hardware limit

    def test_descriptor_json_roundtrip(self, box_hw):
        assert HardwareDescriptor.from_json(box_hw.to_json()) == box_hw


# ------------------------------------------------------------- model shape

# glm5_next (GLM-5.3-class) model section: 45 hidden layers, 3 dense, one
# NextN/MTP block, 288 routed experts — the shape the MTP expert census
# extends (P-29 step 13).
_GLM5_NEXT_MODEL = {
    "architecture": "glm5_next", "num_hidden_layers": 45,
    "hidden_size": 4096, "num_attention_heads": 64,
    "num_key_value_heads": 64, "kv_lora_rank": 512,
    "qk_rope_head_dim": 0, "qk_nope_head_dim": 256, "v_head_dim": 256,
    "q_lora_rank": 1536, "intermediate_size": 12288,
    "n_routed_experts": 288, "n_shared_experts": 1,
    "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3, "moe_layer_freq": 1, "vocab_size": 154880,
    "max_position_embeddings": 1048576, "num_nextn_predict_layers": 1,
    "index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128,
    "index_topk_freq": 0, "index_skip_topk_offset": 0, "index_kpool": 4,
}


class TestModelShape:
    def test_glm52_censuses(self):
        s = glm52_shape()
        assert s.num_moe_layers == 75
        assert s.num_kv_layers == 78
        assert s.engine_kv_pool_layers() == 79
        assert s.num_dsa_computing_layers == 21  # IndexShare: 3 leading + every 4th

    def test_glm5_next_like_censuses(self):
        types = tuple("linear_attention" if l % 4 != 3 else "full_attention"
                      for l in range(45))
        s = ModelShape.from_model_section({
            "architecture": "glm5_next", "num_hidden_layers": 45,
            "hidden_size": 4096, "num_attention_heads": 64,
            "num_key_value_heads": 64, "kv_lora_rank": 512,
            "qk_rope_head_dim": 0, "qk_nope_head_dim": 256, "v_head_dim": 256,
            "q_lora_rank": 1536, "intermediate_size": 12288,
            "n_routed_experts": 288, "n_shared_experts": 1,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "first_k_dense_replace": 3, "vocab_size": 154880,
            "max_position_embeddings": 1048576, "num_nextn_predict_layers": 1,
            "index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128,
            "index_topk_freq": 0, "index_skip_topk_offset": 0, "index_kpool": 4,
            "layer_types": types,
            "linear_attn_config": {"num_heads": 64, "head_dim": 128,
                                   "short_conv_kernel_size": 4},
        })
        n_lin = sum(1 for t in types if t == "linear_attention")
        assert s.num_linear_attention_layers == n_lin
        assert s.num_kv_layers == 45 - n_lin
        assert s.has_index_pool
        # computing set = sparse layers + MTP layer (model_config.cpp:102-121)
        assert s.num_dsa_computing_layers == (45 - n_lin) + 1

    def test_mtp_expert_census_is_config_armed(self):
        # P-29 step 13 / TD-MTP-PROBE-DEFERRED-CONSUMERS: the engine arms the MTP
        # expert census from CONFIG (speculation.enabled + method "mtp" +
        # speculation.mtp.enabled on glm5_next, model_config.cpp:25-31), and
        # then the NextN block is a full MoE layer.  Autoconfig must plan
        # the SAME census or it under-funds the host arena by 288 slots.
        m = _GLM5_NEXT_MODEL
        armed_spec = {"enabled": True, "method": "mtp", "mtp": {"enabled": True}}

        unarmed = ModelShape.from_model_section(m)
        assert unarmed.num_moe_layers == 42          # 45 - 3 dense
        assert not unarmed.is_moe_layer(45)
        assert not unarmed.mtp_experts_armed

        armed = ModelShape.from_config({"model": m, "speculation": armed_spec})
        assert armed.mtp_experts_armed
        assert armed.num_moe_layers == 43            # + the NextN block
        assert armed.is_moe_layer(45) and not armed.is_moe_layer(46)
        # what the sizer funds: 12,096 -> 12,384 slots (the engine's number)
        slot = 1 << 20
        assert sizing.host_arena_total_bytes(
            slot, armed.n_routed_experts, armed.num_moe_layers) == slot * 12_384
        assert sizing.host_arena_total_bytes(
            slot, unarmed.n_routed_experts, unarmed.num_moe_layers) == slot * 12_096

    @pytest.mark.parametrize("spec", [
        {},
        {"enabled": False, "method": "mtp", "mtp": {"enabled": True}},
        {"enabled": True, "method": "dspark", "mtp": {"enabled": True}},
        {"enabled": True, "method": "mtp"},
        {"enabled": True, "method": "mtp", "mtp": {"enabled": False}},
    ])
    def test_mtp_census_unarmed_keeps_champion_count(self, spec):
        s = ModelShape.from_config({"model": _GLM5_NEXT_MODEL,
                                    "speculation": spec})
        assert not s.mtp_experts_armed
        assert s.num_moe_layers == 42

    def test_mtp_census_is_glm5_next_only(self):
        # The arch half of the engine's test: a non-glm5_next model never
        # grows MTP expert tenants, however the recipe arms speculation.
        s = ModelShape.from_config({
            "model": dict(_GLM5_NEXT_MODEL, architecture="deepseek_v32"),
            "speculation": {"enabled": True, "method": "mtp",
                            "mtp": {"enabled": True}}})
        assert s.mtp_experts_armed          # config says armed ...
        assert s.num_moe_layers == 42       # ... but the arch does not carry it


# ------------------------------------------------------------- sizing

class TestSizing:
    def test_kv_bytes_per_token_arms(self):
        s = glm52_shape()
        # vram_allocator.h:249-250 asserted constants
        assert sizing.kv_bytes_per_token(s, "fp8_e4m3", "snapmla") == 644
        assert sizing.kv_bytes_per_token(s, "fp8_e4m3", "turboquant_mla") == 386
        assert sizing.kv_bytes_per_page(s, "fp8_e4m3", "turboquant_mla", 16) == 6176
        assert sizing.kv_bytes_per_page(s, "fp8_e4m3", "snapmla", 16) == 10304

    def test_indexer_page_and_slab(self):
        s = glm52_shape()
        assert sizing.indexer_k_bytes_per_page(s, "fp8_e4m3", 8192) == 1081344
        g = sizing.slab_geometry(s, "fp8_e4m3", "snapmla", 16, 8192)
        assert (g.pages_per_slab, g.slab_bytes) == (105, 1081920)  # vram_allocator boot log
        g = sizing.slab_geometry(s, "fp8_e4m3", "turboquant_mla", 16, 8192)
        assert (g.pages_per_slab, g.slab_bytes) == (176, 1086976)

    def test_auto_kv_pages_sharded_rank0_worst_case(self):
        # champion geometry: 2 x 25600 tokens, dcp=2 sharded, chunk 16, 79 layers
        pages = sizing.auto_kv_pages(25600, 16, 2, 16, 2, 79, 100 << 30, 6176)
        assert pages == 2 * 800 * 79  # 1600 pages/seq, rank-0 share 800
        # replicated: no division
        assert sizing.auto_kv_pages(25600, 16, 1, 16, 2, 79, 100 << 30, 6176) == 2 * 1600 * 79
        # VRAM cap binds
        assert sizing.auto_kv_pages(25600, 16, 1, 16, 2, 79, 6176 * 1000, 6176) == 1000

    def test_kda_slot_bytes_glm53_flash(self):
        types = tuple(["linear_attention"] * 34 + ["full_attention"] * 11)
        s = ModelShape.from_model_section({
            "architecture": "glm5_next", "num_hidden_layers": 45,
            "hidden_size": 4096, "num_attention_heads": 64,
            "num_key_value_heads": 64, "kv_lora_rank": 512,
            "qk_rope_head_dim": 0, "qk_nope_head_dim": 256, "v_head_dim": 256,
            "q_lora_rank": 1536, "intermediate_size": 12288,
            "n_routed_experts": 288, "n_shared_experts": 1,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "first_k_dense_replace": 3, "vocab_size": 154880,
            "max_position_embeddings": 1048576, "num_nextn_predict_layers": 1,
            "index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128,
            "index_topk_freq": 0, "index_skip_topk_offset": 0, "index_kpool": 4,
            "layer_types": types,
        })
        # 34 layers x (64*128*128*4 + 3*3*64*128*4) = 152 633 344 (~145.6 MiB;
        # TD-KDA-STATE-MAPPED-SLABS: 40 slots -> ~5.7 GiB)
        slot = sizing.kda_slot_bytes(s, 1)
        assert slot == 152633344
        assert sizing.kda_policy_slots(32, True, 8) == 40
        assert 5.6 < (40 * slot) / (1 << 30) < 5.8
        assert sizing.kda_slot_bytes(glm52_shape(), 2) == 0  # no linear layers

    def test_kv_tiering_champion_pools(self):
        s = glm52_shape()
        kt = sizing.kv_tiering_sizes(s, "fp8_e4m3", "turboquant_mla", 16,
                                     2048, 8.0, 79, 2, kv_sharded=True)
        assert kt.hot_slots == 2048
        # sharded KV: per-rank shard tiering, NO dedup divide — verified vs
        # the 2026-08-30 derived-recipe boot log (80896 pages/rank)
        assert kt.cold_pool_pages == 80896
        # replicated KV at dcp=2: one cold copy per page (INV-KVT-11)
        assert sizing.kv_tiering_sizes(s, "fp8_e4m3", "turboquant_mla", 16,
                                       2048, 8.0, 79, 2,
                                       kv_sharded=False).cold_pool_pages == 40448
        # auto hot = 2*index_topk when hot_buffer_slots == 0
        assert sizing.kv_tiering_sizes(s, "fp8_e4m3", "turboquant_mla", 16,
                                       0, 8.0, 79, 2).hot_slots == 4096

    def test_expert_slots(self):
        assert sizing.expert_slots_in_zone(4 << 30, 27623424) == 155
        assert sizing.expert_slots_in_zone(int(11.5 * (1 << 30)), 27623424) == 447
        assert sizing.host_arena_total_bytes(27623424, 256, 75) == 530369740800

    def test_v4_tier_demand_smoke(self):
        ratios = [0, 0] + [4, 128] * 20 + [0]
        s = ModelShape.from_model_section({
            "architecture": "deepseek_v4", "num_hidden_layers": 43,
            "hidden_size": 4096, "num_attention_heads": 64,
            "num_key_value_heads": 1, "kv_lora_rank": 512,
            "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
            "q_lora_rank": 1024, "intermediate_size": 12288,
            "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
            "first_k_dense_replace": 0, "vocab_size": 129280,
            "max_position_embeddings": 1048576, "num_nextn_predict_layers": 0,
            "index_topk": 512, "index_n_heads": 64, "index_head_dim": 128,
            "index_topk_freq": 0, "index_skip_topk_offset": 0, "index_kpool": 1,
            "compress_ratios": ratios, "sliding_window": 128, "o_groups": 8,
        })
        d = sizing.v4_tier_demand(s, "csa_hca", 25600, 2, 8, 25600, 8192, 0.15)
        assert d.csa_pages == 2 * 100 * 20      # not holder-scaled
        assert d.hca_pages == (2 + 8) * 100 * 20  # holder-scaled
        assert d.csa_bytes_per_page == 1160 * 64
        assert d.hca_bytes_per_page == 1160 * 2
        tq = sizing.v4_tier_demand(s, "csa_hca_tq_mix", 25600, 2, 8, 25600, 8192, 0.15)
        assert tq.csa_bytes_per_page == 644 * 64
        assert tq.hca_bytes_per_page == 1160 * 2  # mix: HCA stays FP8


# ------------------------------------------------------------- calibration

def fake_calibration(tmp_path, uuids=None, n=4096, k=6144):
    uuids = uuids or [BOX_GPUS[i][2][:39] for i in (2, 3, 0, 1)]  # engine order, truncated
    numa = [2, 3, 0, 2]
    names = ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5090",
             "NVIDIA GeForce RTX 5080", "NVIDIA GeForce RTX 5080"]
    a_us = [13.83, 13.84, 24.70, 24.68]
    d = {
        "version": 2, "source": "calibrated", "expert_bytes": 24772992.0,
        "num_devices": 4, "num_banks": 4, "compute_N": n, "compute_K": k,
        "compute_tokens": 1, "recon_payload_bytes": 12288.0,
        "fixed_overhead_us": 0.0, "ncf": [0.0, 1.0, 1.34],
        "devices": [
            {"position": i, "numa_node": numa[i], "name": names[i],
             "uuid": uuids[i], "xfer_lat_us": 1.5,
             "compute": {"a_us": a_us[i], "b_us": 60.0, "P": 64}}
            for i in range(4)
        ],
        "banks": [{"node": b, "egress_us": 440.0, "contention": 0.4}
                  for b in range(4)],
        "matrix": [[{"rate_us": 500.0 + 10 * b + d_, "tier": 1 if (b, d_) in
                     ((0, 2), (2, 0), (2, 3), (3, 1)) else 2, "lat_us": 1.5}
                    for d_ in range(4)] for b in range(4)],
    }
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(d))
    return str(p)


class TestCalibration:
    ORDER = [2, 3, 0, 1]  # hardware.gpus emission order as detection ordinals

    def test_load_and_accept(self, box_hw, tmp_path):
        cal = load_calibration(fake_calibration(tmp_path))
        assert accept_calibration(cal, box_hw, self.ORDER, 6144, 2048) == []

    def test_reject_wrong_machine(self, box_hw, tmp_path):
        cal = load_calibration(fake_calibration(
            tmp_path, uuids=["GPU-dead" + "0" * 32] * 4))
        reasons = accept_calibration(cal, box_hw, self.ORDER, 6144, 2048)
        assert any("INV-LOADER-CAL-6" in r for r in reasons)

    def test_reject_wrong_model_dims(self, box_hw, tmp_path):
        cal = load_calibration(fake_calibration(tmp_path))
        reasons = accept_calibration(cal, box_hw, self.ORDER, 7168, 2048)
        assert any("compute_K" in r for r in reasons)

    def test_speed_rank_is_measured_not_name(self, box_hw, tmp_path):
        cal = load_calibration(fake_calibration(tmp_path))
        rank = cal.device_speed_rank()
        assert rank[:2] == [0, 1]  # measured a_us puts the 5090 positions first

    def test_resolve_path(self):
        assert resolve_calibration_path("", "/w/m.gguf") == "/w/gpu_loader_calibration.json"
        assert resolve_calibration_path("cal.json", "/w/m.gguf") == "/w/cal.json"
        assert resolve_calibration_path("/abs/c.json", "/w/m.gguf") == "/abs/c.json"
        # relative-WITH-dir also joins the weights dir (live-boot verified)
        assert resolve_calibration_path("sub/c.json", "/w/m.gguf") == "/w/sub/c.json"


# ------------------------------------------------------------- fingerprint

class TestFingerprint:
    def test_stable_and_sensitive(self, box_hw, tmp_path):
        cal = load_calibration(fake_calibration(tmp_path))
        fp1 = fingerprint(box_hw, cal)
        assert fp1.startswith("hwfp1-")
        # stable: rates are volatile and excluded
        d = json.loads(open(fake_calibration(tmp_path)).read())
        for row in d["matrix"]:
            for c in row:
                c["rate_us"] *= 1.5
        p = tmp_path / "cal2.json"
        p.write_text(json.dumps(d))
        assert fingerprint(box_hw, load_calibration(str(p))) == fp1
        # sensitive: a re-cabling flips a tier
        d["matrix"][0][0]["tier"] = 1
        p.write_text(json.dumps(d))
        assert fingerprint(box_hw, load_calibration(str(p))) != fp1

    def test_sensitive_to_reslotting(self, tmp_path):
        proc, sysd = build_fake_tree(str(tmp_path / "a"))
        hw1 = detect_hardware(proc_root=proc, sys_root=sysd)
        proc, sysd = build_fake_tree(str(tmp_path / "b"), pcie_max="16.0 GT/s PCIe")
        hw2 = detect_hardware(proc_root=proc, sys_root=sysd)
        assert fingerprint(hw1) != fingerprint(hw2)  # Gen5 slot vs Gen4 slot


# ------------------------------------------------------------- constraints

class TestEngineConstraints:
    def test_rows_have_provenance(self):
        for row in ENGINE_CONSTRAINTS:
            assert row.kind in ("engine_gate", "measured", "validator")
            assert row.site, f"{row.id} lacks a citation"

    def test_scope_filter(self):
        ids = {r.id for r in active_for("glm_moe_dsa")}
        # local-indexer-disables-tiering was DELETED when
        # TD-KVT-LOCAL-INDEXER-UNBLOCK resolved (2026-08-30, engine gate
        # removed — tiering composes with dcp_indexer_mode=local).
        assert "local-indexer-disables-tiering" not in ids
        assert "dspark-ctx-cap" in ids
        assert "v4-no-sharded-kv" not in ids
        assert "kda-state-fixed-carve" not in ids
        assert get("dspark-ctx-cap").ticket == "TD-DSPARK-CTX-POLICY"

    def test_the_mapped_kda_row_replaced_the_dead_carve_row(self):
        """TD-AUTOCONFIG-STALE-KDA-CARVE-ROW: `kda-state-fixed-carve`
        outlived the ticket that retired it (TD-KDA-STATE-MAPPED-SLABS made
        mapped the default on 2026-08-31) and went on feeding the solver a
        dedicated-carve model of a SHARED-pool tenant. A dead row must stay
        dead — the registry's whole value is that a row present means the
        rule holds — so both directions are pinned: the carve row is gone
        for glm5_next too, and the row that replaced it is active there."""
        ids = {r.id for r in active_for("glm5_next")}
        assert "kda-state-mapped-tenant" in ids
        assert "kda-state-fixed-carve" not in ids
        assert get("kda-state-fixed-carve") is None
        row = get("kda-state-mapped-tenant")
        assert row.ticket == "TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA"
        assert row.kind == "engine_gate" and row.scope == "glm5_next"
        # scoped, so the MLA+DSA champion arch never sees it
        assert "kda-state-mapped-tenant" not in {r.id
                                                 for r in active_for("glm_moe_dsa")}
        # the consequence the solver reads off it: one pool, three tenants
        assert "SLAB-PADDED" in row.statement
        assert "kv_main" in row.statement


# ------------------------------------------------------------- gguf meta

def _gguf_kv(f, key, value):
    """One metadata KV. Types: str(8), u32(4), f32(6), bool(7), i32(5),
    array(9) of u32 or f32 (a float element anywhere makes it an f32
    array — glm5next publishes swiglu_clamp_exp that way)."""
    kb = key.encode()
    f.write(struct.pack("<Q", len(kb)))
    f.write(kb)
    if isinstance(value, bool):
        f.write(struct.pack("<I", 7))
        f.write(struct.pack("<?", value))
    elif isinstance(value, int):
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<I", value))
    elif isinstance(value, float):
        f.write(struct.pack("<I", 6))
        f.write(struct.pack("<f", value))
    elif isinstance(value, str):
        vb = value.encode()
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(vb)))
        f.write(vb)
    elif isinstance(value, list):   # array of u32 (or f32)
        floats = any(isinstance(v, float) for v in value)
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 6 if floats else 4))
        f.write(struct.pack("<Q", len(value)))
        for v in value:
            f.write(struct.pack("<f", float(v)) if floats
                    else struct.pack("<I", int(v)))
    else:
        raise TypeError(type(value))


def write_fake_gguf(path, tensors, kv=None):
    """Minimal GGUF v3 writer: metadata KVs (optional) + tensor infos."""
    kv = kv or {}
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kv)))
        for k, v in kv.items():
            _gguf_kv(f, k, v)
        for name, dims, gtype in tensors:
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack(f"<{len(dims)}Q", *dims))
            f.write(struct.pack("<I", gtype))
            f.write(struct.pack("<Q", 0))


class TestGgufMeta:
    def test_gg9_per_projection_max(self, tmp_path):
        # layer 0: q4_k everywhere; layer 1: q5_k gate/up, q6_k down (the XL mix)
        p = tmp_path / "m.gguf"
        write_fake_gguf(str(p), [
            ("blk.0.ffn_gate_exps.weight", (6144, 2048, 256), 12),
            ("blk.0.ffn_up_exps.weight", (6144, 2048, 256), 12),
            ("blk.0.ffn_down_exps.weight", (2048, 6144, 256), 12),
            ("blk.1.ffn_gate_exps.weight", (6144, 2048, 256), 13),
            ("blk.1.ffn_up_exps.weight", (6144, 2048, 256), 13),
            ("blk.1.ffn_down_exps.weight", (2048, 6144, 256), 14),
        ])
        total, detail = gguf_meta.expert_slot_bytes_from_gguf(str(p))
        assert detail["gate"] == {"bytes": 8650752, "type": "q5_k"}
        assert detail["up"] == {"bytes": 8650752, "type": "q5_k"}
        assert detail["down"] == {"bytes": 10321920, "type": "q6_k"}
        assert total == 27623424  # matches the GLM-5.2 prepack manifest slot

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REPO, "test-data/GLM-5.2-prepacked/manifest.json")),
        reason="machine-data-gated: GLM-5.2 prepack absent")
    def test_manifest_slot(self):
        assert gguf_meta.expert_slot_bytes_from_manifest(
            os.path.join(REPO, "test-data/GLM-5.2-prepacked")) == 27623424
