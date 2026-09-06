"""Architecture-family serving templates (AUTOCONFIG §5 M1) — DATA.

These are the champion-proven values for knobs that are neither derivable
from hardware/model geometry nor schema defaults: orchestration weights,
prefetch fusion weights, speculation calibration constants, transfer plumb-
ing. The solver COPIES them (kind=template) — its value is knowing WHICH
template applies and where a knob must deviate, not re-judging measured
serving policy. Provenance: recipes/glm52_serve_champion.json (the product
of the 2026-06..08 measured campaigns) and recipes/deepseek_v4_serve_tp2ep4.

Keep entries that duplicate a generated-parser default anyway: a recipe
should be READABLE stand-alone, like the champion is.
"""

# Family key: "mla_dsa" covers glm_moe_dsa / deepseek_v3_2-class (MLA + DSA
# + MoE); "v4" covers deepseek_v4 (CSA/HCA side tiers); "glm5_next" the
# hybrid (KDA + sparse MLA) — starts from mla_dsa with arch overrides.

COMMON = {
    "orchestrator": {
        "performance_objective": {"latency_weight": 0.6, "throughput_weight": 0.4},
        "max_batch_size": 64,
        "expert_batching": {"enabled": True, "max_wait_us": 50,
                            "min_batch_tokens_per_expert": 4},
        "coactivation_graph": {"enabled": True, "decay_factor": 0.999,
                               "reoptimize_interval_seconds": 300,
                               "workload_shift_decay": 0.1},
        "workload_detection": {"ewma_alpha": 0.01, "shift_threshold_std_devs": 3.0},
    },
    "prefetch": {
        "prescope": {"enabled": True, "lookahead_layers": 1},
        "probe": {"enabled": True, "probe_points": [0.25, 0.5, 0.75],
                  "confidence_threshold": 0.6},
        "moe_speq": {"enabled": True, "predictor_hidden_size": 256,
                     "learning_rate": 0.0001, "training_buffer_size": 10000,
                     "train_interval_steps": 100},
        "speculative": {"enabled": True, "max_prefetch_depth_layers": 3,
                        "confidence_threshold": 0.4},
        "fusion_weights": {"prescope_alpha": 0.5, "probe_beta": 0.3,
                           "speq_gamma": 0.2, "sp_moe_delta": 0.15},
    },
    "speculation_scaffold": {
        # everything but the method decision (solver stage P6 owns that)
        "enabled": True,
        "mtp": {"enabled": False, "max_depth": 3, "dynamic_depth": True},
        "self_speculative": {"enabled": False, "draft_expert_count": 1,
                             "adaptive_exit_enabled": True,
                             "draft_confidence_threshold": 0.4,
                             "residual_correction": {"enabled": True, "hidden_size": 128,
                                                     "learning_rate": 5e-05,
                                                     "training_buffer_size": 50000}},
        "prompt_lookup": {"enabled": True, "max_ngram_size": 4,
                          "max_continuation_length": 5},
        "verification": {"moe_spec_max_loaded_experts_fraction": 0.8,
                         "adaptive_topk_threshold": 0.92,
                         "verification_quality_floor": 0.85,
                         "in_flight_transfer_mode": "conservative",
                         "sparse_verification": False,
                         "ranking_strategy": "router_based",
                         "substitution_policy": "substitution"},
        "layer_skip_draft": {"enabled": False, "method": "cosine_similarity",
                             "threshold": 0.995,
                             "min_acceptance_rate_to_enable": 0.5},
        "reasoning_mode": {"enabled": False, "think_token_detection": True,
                           "aggressive_speculation_depth_multiplier": 2.0,
                           "relaxed_verification_threshold": 0.85},
        "calibration": {"min_acceptance_rate": 0.3, "target_acceptance_rate": 0.6,
                        "adjustment_interval_tokens": 500, "acceptance_ema_alpha": 0.5,
                        "draft_count_ema_alpha": 0.9, "adjustment_step_size": 0.01},
        "utility_scorer": {"enabled": True, "utility_threshold": 1.0,
                           "vram_headroom_fraction": 0.1},
    },
    "parallelism": {
        "expert_affinity": {"mode": "soft", "initial_assignment": "round_robin",
                            "rebalance_interval_seconds": 300},
        "expert_duplication": {"enabled": True, "max_duplicated_fraction": 0.05},
        "node_routing": {"max_nodes_per_token": 4},
    },
    "transfer": {
        "fine_grained": {"enabled": True, "max_in_flight_per_gpu": 4,
                         "prefetch_distance_layers": 4},
        "streams_per_gpu": 2,
        "priority_levels": 3,
    },
    "compute_cuda_graphs": {"enabled": True, "capture_attention": True,
                            "capture_gating": True, "capture_tp_allreduce": True,
                            "capture_expert_ffn": False},
    "compute_gemm": {"backend": "cutlass", "grouped_gemm": True, "fused_swiglu": True},
    "expert_cache": {
        "eviction_policy": "impact_weighted_lru",
        "eviction_alpha_recency": 0.4,
        "eviction_beta_frequency": 0.35,
        "eviction_gamma_routing_weight": 0.25,
        "duplication_frequency_threshold_percentile": 95,
        "temporal_autocorrelation_bonus": 0.8,
        "stable_zone_fraction": 0.5,
    },
    "numa": {"interleave_strategy": "affinity_based", "fallback": "round_robin"},
}

# Attention backend by family — measured rows, not free choices:
# mla_dsa: direct-TQ sparse is the ONLY B=1 sparse decode path (TQ campaign,
#          §12l-§12m); KV x0.599 vs snapmla (386 vs 644 B/row at rope 64).
# v4:      csa_hca_tq_mix (TQ-for-V4 via codec composition, goldens 6/6,
#          KV x0.555, SEQSTATE campaign 2026-08-23).
# glm5_next: snapmla — THE CHAMPION SWITCHED (P-29 step 14, 2026-09-05,
#          pre-committed OQ-6/OQ-7 rule, user-granted 2026-09-04): on the
#          fixed post-OQ-8 binary snapmla+FP8-decode beats TQ+split-KV on
#          BOTH axes at the 8k anchor (24.10/24.21/24.04 vs
#          23.67/23.80/23.74 tok/s repeat medians; TF-NLL 1.7897 vs
#          1.8016, -0.0118 +- 0.0060 nats).  `standard` = the champion-
#          proven choice, so the template follows the champion
#          (P-31 step 1 sync; filed as a default change in LOG.md).
#          turboquant_mla remains serveable and is the `compact` tier
#          (KV x0.50: 258 vs 516 B/row at the NoPE geometry; verified
#          live tp=1 2026-09-01, TD-GLM5-TQ-BACKEND-UNWIRED resolved).
FAMILY = {
    "mla_dsa": {
        "attention_backend": "turboquant_mla",
        "dsa_sparse_prefill": True,
        "stable_zone_fraction_per_gpu": 0.95,
    },
    "v4": {
        "attention_backend": "csa_hca_tq_mix",
        "dsa_sparse_prefill": True,
        "stable_zone_fraction_per_gpu": 0.95,
    },
    "glm5_next": {
        "attention_backend": "snapmla",
        "dsa_sparse_prefill": True,
        "stable_zone_fraction_per_gpu": 0.95,
        # GF3 serving recipe values (recipes/glm53flash_serve.json) — the
        # TD-AUTOCONFIG-NO-SERVING-SURFACE resolution: without these an
        # autoconfigured glm5_next boot silently loses everything GF3.13
        # shipped (reasoning split, tool-call wire format, glm5_next
        # tokenizer mode incl. reasoning_effort normalization and the
        # stop-token autodetect fix) and GF3.15's measured superchunk win.
        "max_batch_size": 512,
        "prefill_superchunk_tokens": 512,
        "serving_surface": {
            "tokenizer_mode": "glm5_next",
            "tool_call_parser": "glm47",
            "reasoning_parser": "glm45",
            "enable_auto_tool_choice": True,
        },
    },
}


# Accuracy ladder per family (TD-AUTOCONFIG-ACCURACY-LEVER): every backend
# the engine can actually SERVE for the family (config_validator.cpp:510-534
# partition — {snapmla, turboquant_mla} for the MLA families, {csa_hca,
# csa_hca_tq, csa_hca_tq_mix} for V4), ordered MOST-ACCURATE-FIRST.  The
# order is not a weight: within a family each step down quantizes strictly
# more of the KV path to the 4-bit TQ codec (kv_codec.h composition axis —
# MLA: kFp8 -> kTq4; V4 tiers: kFp8/kFp8 -> kTq4/kFp8 -> kTq4/kTq4), so the
# precision ordering is structural; the MLA step is additionally MEASURED
# (registry row `mla-tq-vs-snapmla-accuracy`).  The `accuracy` lever picks a
# floor over this ladder; `standard` is the family template's champion-proven
# choice.  A hypothetical full-precision arm (KV bf16, codec kFull) is NOT on
# any ladder because the engine does not build one (kv_codec.h:12-14) — that
# is the `superior` tier, defined but refused.
ACCURACY_LADDER = {
    "mla_dsa": ("snapmla", "turboquant_mla"),
    "glm5_next": ("snapmla", "turboquant_mla"),
    "v4": ("csa_hca", "csa_hca_tq_mix", "csa_hca_tq"),
}


def family_of(architecture: str) -> str:
    if architecture == "deepseek_v4":
        return "v4"
    if architecture == "glm5_next":
        return "glm5_next"
    return "mla_dsa"


def template_view(architecture: str) -> dict:
    """Recipe-coordinate view of the template DATA the solver copies
    verbatim in Solver._assemble (TD-AUTOCONFIG-COMPARE-CLASSES): the
    compare classifier uses it to tell a family-template copy with no
    Explanation row apart from a real solver gap.  Keep the mapping in
    lock-step with _assemble's `dict(T[...])` copies — decision fields the
    solver overwrites (speculation.method/enabled/dspark,
    parallelism.tensor_parallelism, orchestrator rows with their own
    Explanations) may appear here; their Explanation rows take precedence
    in the classifier, so listing them is harmless."""
    fam = family_of(architecture)
    ftmpl = FAMILY[fam]
    orch = dict(COMMON["orchestrator"])
    if "max_batch_size" in ftmpl:
        orch["max_batch_size"] = ftmpl["max_batch_size"]
    compute = {
        "cuda_graphs": COMMON["compute_cuda_graphs"],
        "gemm": COMMON["compute_gemm"],
        "attention_backend": ftmpl["attention_backend"],
        "dsa_sparse_prefill": ftmpl["dsa_sparse_prefill"],
    }
    if "prefill_superchunk_tokens" in ftmpl:
        compute["prefill_superchunk_tokens"] = ftmpl["prefill_superchunk_tokens"]
    return {
        "orchestrator": orch,
        "prefetch": COMMON["prefetch"],
        "speculation": COMMON["speculation_scaffold"],
        "parallelism": COMMON["parallelism"],
        "transfer": COMMON["transfer"],
        "compute": compute,
        "memory": {
            "expert_cache": COMMON["expert_cache"],
            "numa": COMMON["numa"],
        },
        "serving": dict(ftmpl.get("serving_surface") or {}),
    }
