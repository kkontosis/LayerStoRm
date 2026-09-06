"""Config dataclasses for the autoconfig sections (CLAUDE.md Python config
convention: every orchestrator config dataclass has a matching JSON schema
section).

Schema sources:
  ``autoconfig``            — config/schema.json (user-facing: the opt-in
                              switch, the two levers, output path, fingerprint)
  ``_internal-autoconfig``  — config/internal-schemas/autoconfig.schema.json
                              (solver policy knobs; x-configFile autoconfig.json)

Defaults here, in the schema, and in the readers must agree (the
PrefixCacheConfig.from_config house pattern).
"""

from __future__ import annotations

from dataclasses import dataclass

from .solver import AutoconfigKnobs


@dataclass(frozen=True)
class AutoconfigConfig:
    """autoconfig section (config/schema.json)."""
    enabled: bool = False
    vram_expert_ratio: float | None = None
    total_active_context_tokens: int | None = None
    output_path: str = ""
    fingerprint: str = ""

    @classmethod
    def from_config(cls, cfg: dict) -> "AutoconfigConfig":
        ac = cfg.get("autoconfig") or {}
        ratio = ac.get("vram_expert_ratio")
        ctx = ac.get("total_active_context_tokens")
        return cls(
            enabled=bool(ac.get("enabled", False)),
            vram_expert_ratio=float(ratio) if ratio is not None else None,
            total_active_context_tokens=int(ctx) if ctx is not None else None,
            output_path=str(ac.get("output_path", "")),
            fingerprint=str(ac.get("fingerprint", "")),
        )


def knobs_from_config(cfg: dict) -> AutoconfigKnobs:
    """_internal-autoconfig -> AutoconfigKnobs (solver policy). Fields not in
    the internal schema keep their dataclass defaults."""
    ia = cfg.get("_internal-autoconfig") or {}
    base = AutoconfigKnobs()
    return AutoconfigKnobs(
        vram_usable_fraction=float(ia.get("vram_usable_fraction",
                                          base.vram_usable_fraction)),
        vram_safety_margin_gb=float(ia.get("vram_safety_margin_gb",
                                           base.vram_safety_margin_gb)),
        non_tp_overhead_gib=float(ia.get("non_tp_overhead_gib",
                                         base.non_tp_overhead_gib)),
        host_pin_fraction_total=float(ia.get("host_pin_fraction_total",
                                             base.host_pin_fraction_total)),
        hbm_spill_fraction_free=float(ia.get("hbm_spill_fraction_free",
                                             base.hbm_spill_fraction_free)),
        prefill_scratch_gb=float(ia.get("prefill_scratch_gb",
                                        base.prefill_scratch_gb)),
        min_seq_for_pairing=int(ia.get("min_seq_for_pairing",
                                       base.min_seq_for_pairing)),
        device_pool_round_pages=int(ia.get("device_pool_round_pages",
                                           base.device_pool_round_pages)),
        expert_gib_quantum=float(ia.get("expert_gib_quantum",
                                        base.expert_gib_quantum)),
        min_max_seq=int(ia.get("min_max_seq", base.min_max_seq)),
        moe_fit_fixed_attn_mib=int(ia.get("moe_fit_fixed_attn_mib",
                                          base.moe_fit_fixed_attn_mib)),
        moe_fit_fixed_expert_mib=int(ia.get("moe_fit_fixed_expert_mib",
                                            base.moe_fit_fixed_expert_mib)),
        moe_fit_headroom_cushion=float(ia.get(
            "moe_fit_headroom_cushion", base.moe_fit_headroom_cushion)),
        moe_fit_free_floor_mib=int(ia.get("moe_fit_free_floor_mib",
                                          base.moe_fit_free_floor_mib)),
        runtime_margin_quantum_gb=float(ia.get(
            "runtime_margin_quantum_gb", base.runtime_margin_quantum_gb)),
        stride_margin_cap_gb=float(ia.get("stride_margin_cap_gb",
                                          base.stride_margin_cap_gb)),
        block_table_budget_gb=float(ia.get("block_table_budget_gb",
                                           base.block_table_budget_gb)),
    )
