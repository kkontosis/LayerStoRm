"""Model shape ingestion for autoconfig (AUTOCONFIG §3).

Derives the sizing-relevant geometry from a recipe ``model`` section (the
canonical form — field names match DeepSeek HF config naming per CLAUDE.md),
or from an HF ``config.json`` for the known architecture families.

The layer-census logic here is a transcription of
``src/model/model_config.cpp`` (``computes_indexer`` / ``is_full_index_layer``
/ ``compute_layer_counts``); comments cite the C++ site. Divergence between
this file and model_config.cpp is a bug HERE.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinearAttnGeometry:
    """model.linear_attn_config (glm5_next KDA layers)."""
    num_heads: int = 64
    head_dim: int = 128
    short_conv_kernel_size: int = 4


@dataclass(frozen=True)
class ModelShape:
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    q_lora_rank: int
    intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    first_k_dense_replace: int
    moe_layer_freq: int
    vocab_size: int
    max_position_embeddings: int
    num_nextn_predict_layers: int
    # DSA / indexer
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    index_topk_freq: int
    index_skip_topk_offset: int
    index_kpool: int
    # hybrid (glm5_next)
    layer_types: tuple[str, ...] = ()
    linear_attn: LinearAttnGeometry | None = None
    # V4
    compress_ratios: tuple[int, ...] = ()
    sliding_window: int = 0
    o_groups: int = 1
    # P-29 step 13 / TD-MTP-PROBE-DEFERRED-CONSUMERS: mirrors the engine's
    # config-armed MTP expert census (model_config.cpp ModelConfig ctor →
    # arm_mtp_experts).  True = the recipe arms MTP drafting
    # (speculation.enabled + speculation.method "mtp" +
    # speculation.mtp.enabled) on a glm5_next model, so the NextN block's
    # routed experts are arena tenants and MUST be counted here — planning
    # the un-extended census under-funds the host arena by one MoE layer
    # (glm5_next: 12,096 planned vs 12,384 allocated slots).
    mtp_experts_armed: bool = False
    raw: dict = field(default_factory=dict, hash=False, compare=False)

    # -- family predicates (model_config.cpp) --
    @property
    def is_v4(self) -> bool:
        return self.architecture == "deepseek_v4"

    @property
    def is_glm5_next(self) -> bool:
        return self.architecture == "glm5_next"

    @property
    def has_dsa(self) -> bool:
        return self.index_topk > 0

    @property
    def has_index_pool(self) -> bool:
        # model_config.cpp:46 — glm5_next only
        return self.is_glm5_next and self.index_kpool > 1

    @property
    def uses_mla(self) -> bool:
        return not self.is_v4

    def is_linear_attention_layer(self, l: int) -> bool:
        if l < 0 or l >= len(self.layer_types):
            return False
        return self.layer_types[l] == "linear_attention"

    # -- layer censuses (model_config.cpp compute_layer_counts) --
    def is_moe_layer(self, l: int) -> bool:
        if l < self.first_k_dense_replace:
            return False
        if l >= self.num_hidden_layers:
            # model_config.cpp:41-52 — the glm5_next MTP/NextN block(s) are
            # full MoE layers (288 routed + shared + gate).  Counted ONLY
            # when the census is armed, exactly as the engine does: armed
            # config + glm5_next + inside the NextN range.  Unarmed the
            # bound is the historical `False`, so champion layer counts and
            # the 12,096-slot arena identity are untouched.
            return (self.mtp_experts_armed and self.is_glm5_next
                    and l < self.num_hidden_layers
                    + self.num_nextn_predict_layers)
        freq = max(1, self.moe_layer_freq)
        return (l - self.first_k_dense_replace) % freq == 0

    @property
    def num_moe_layers(self) -> int:
        # Extended range (model_config.cpp compute_layer_counts appends the
        # NextN MoE layers after the hidden-layer sweep); is_moe_layer is
        # False past num_hidden_layers unless the census is armed.
        return sum(1 for l in range(self.num_hidden_layers
                                    + self.num_nextn_predict_layers)
                   if self.is_moe_layer(l))

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for l in range(self.num_hidden_layers)
                   if self.is_linear_attention_layer(l))

    @property
    def num_kv_layers(self) -> int:
        """KV-BEARING hidden layers (model_config.cpp:166)."""
        return self.num_hidden_layers - self.num_linear_attention_layers

    def engine_kv_pool_layers(self) -> int:
        """What the engine's KV-pool sizer uses (vram_allocator.cpp:417,
        TD-KV-POOL-SIZED-OVER-ALL-LAYERS FIXED 2026-08-30): KV-BEARING
        layers + nextn. Uniform-attention models: == hidden + nextn by
        construction (byte-identical substitution); hybrids: linear layers
        carry recurrent state, not KV pages."""
        return self.num_kv_layers + self.num_nextn_predict_layers

    # -- indexer census (model_config.cpp:89-121) --
    def is_full_index_layer(self, l: int) -> bool:
        if l < 0 or l >= self.num_hidden_layers:
            return False
        if self.is_glm5_next:
            return not self.is_linear_attention_layer(l)
        if self.index_topk_freq <= 0:
            return True
        if l < self.index_skip_topk_offset:
            return True
        return (l - self.index_skip_topk_offset + 1) % self.index_topk_freq == 0

    def computes_indexer(self, l: int) -> bool:
        if not self.has_dsa:
            return False
        nh = self.num_hidden_layers
        if l < 0 or l >= nh + self.num_nextn_predict_layers:
            return False
        if l >= nh:  # MTP layer(s)
            return self.is_glm5_next
        if self.is_glm5_next:
            return not self.is_linear_attention_layer(l)
        return self.is_full_index_layer(l) or l == 0

    @property
    def num_dsa_computing_layers(self) -> int:
        """vram_allocator.cpp:478-484 — GLM-5.2: 21 of 79."""
        return sum(1 for l in range(self.num_hidden_layers + self.num_nextn_predict_layers)
                   if self.computes_indexer(l))

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelShape":
        """Shape from a WHOLE recipe — the form that can see the
        speculation section, and therefore the only one that can resolve
        the MTP expert census (P-29 step 13).  Prefer this over
        ``from_model_section`` wherever the full config is in hand."""
        return cls.from_model_section(
            (cfg or {}).get("model") or {},
            mtp_experts_armed=mtp_experts_armed(cfg))

    @classmethod
    def from_model_section(cls, m: dict,
                           *, mtp_experts_armed: bool = False) -> "ModelShape":
        lin = None
        if m.get("linear_attn_config"):
            la = m["linear_attn_config"]
            lin = LinearAttnGeometry(
                num_heads=int(la.get("num_heads", 64)),
                head_dim=int(la.get("head_dim", 128)),
                short_conv_kernel_size=int(la.get("short_conv_kernel_size", 4)),
            )
        return cls(
            architecture=str(m.get("architecture", "")),
            num_hidden_layers=int(m.get("num_hidden_layers", 0)),
            hidden_size=int(m.get("hidden_size", 0)),
            num_attention_heads=int(m.get("num_attention_heads", 0)),
            num_key_value_heads=int(m.get("num_key_value_heads", 0)),
            kv_lora_rank=int(m.get("kv_lora_rank", 0)),
            qk_rope_head_dim=int(m.get("qk_rope_head_dim", 0)),
            qk_nope_head_dim=int(m.get("qk_nope_head_dim", 0)),
            v_head_dim=int(m.get("v_head_dim", 0)),
            q_lora_rank=int(m.get("q_lora_rank", 0)),
            intermediate_size=int(m.get("intermediate_size", 0)),
            n_routed_experts=int(m.get("n_routed_experts", 0)),
            n_shared_experts=int(m.get("n_shared_experts", 0)),
            num_experts_per_tok=int(m.get("num_experts_per_tok", 0)),
            moe_intermediate_size=int(m.get("moe_intermediate_size", 0)),
            first_k_dense_replace=int(m.get("first_k_dense_replace", 0)),
            moe_layer_freq=int(m.get("moe_layer_freq", 1)),
            vocab_size=int(m.get("vocab_size", 0)),
            max_position_embeddings=int(m.get("max_position_embeddings", 0)),
            num_nextn_predict_layers=int(m.get("num_nextn_predict_layers", 0)),
            index_topk=int(m.get("index_topk", 0)),
            index_n_heads=int(m.get("index_n_heads", 0)),
            index_head_dim=int(m.get("index_head_dim", 0)),
            index_topk_freq=int(m.get("index_topk_freq", 0)),
            index_skip_topk_offset=int(m.get("index_skip_topk_offset", 0)),
            index_kpool=int(m.get("index_kpool", 1)),
            layer_types=tuple(m.get("layer_types", ()) or ()),
            linear_attn=lin,
            compress_ratios=tuple(m.get("compress_ratios", ()) or ()),
            sliding_window=int(m.get("sliding_window", 0)),
            o_groups=int(m.get("o_groups", 1)),
            mtp_experts_armed=bool(mtp_experts_armed),
            raw=dict(m),
        )


def mtp_experts_armed(cfg: dict) -> bool:
    """Does this recipe arm the engine's MTP expert census?

    Transcription of the ModelConfig constructor's arming test
    (src/model/model_config.cpp:25-31): speculation on, method "mtp", the
    mtp sub-section enabled.  The architecture / nextn-depth half of the
    engine's test lives in ``ModelShape.is_moe_layer`` (it needs the model
    section).  Autoconfig cannot read the engine's env latch (LS_MTP_PROBE)
    — CONFIG is the only arming signal it may plan from.
    """
    spec = (cfg or {}).get("speculation") or {}
    if not spec.get("enabled"):
        return False
    if str(spec.get("method", "")) != "mtp":
        return False
    return bool((spec.get("mtp") or {}).get("enabled"))
