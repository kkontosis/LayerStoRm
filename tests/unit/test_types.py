"""Tests for orchestrator.types — dataclass construction, ExpertKey hashing, enum interop."""

import pytest

from orchestrator.types import (
    AffinityHint,
    CacheZone,
    EngineMetadata,
    ExpertKey,
    GpuConfig,
    GpuTier,
    HostTier,
    PrefetchConfidence,
    PrefetchHint,
    PrefetchSource,
    WorkItemExperts,
)
from orchestrator.shm_protocol import (
    GPU_TIER_ABSENT,
    GPU_TIER_HOT,
    GPU_TIER_TRANSFERRING,
    HOST_TIER_COLD,
    HOST_TIER_LOADING,
    HOST_TIER_WARM,
    ZONE_STABLE,
    ZONE_STREAMING,
)


class TestExpertKey:
    def test_construction(self):
        k = ExpertKey(layer_idx=3, expert_idx=42)
        assert k.layer_idx == 3
        assert k.expert_idx == 42

    def test_positional_construction(self):
        k = ExpertKey(3, 42)
        assert k.layer_idx == 3
        assert k.expert_idx == 42

    def test_hashing_and_dict_key(self):
        k1 = ExpertKey(1, 5)
        k2 = ExpertKey(1, 5)
        d = {k1: "value"}
        assert d[k2] == "value"
        assert hash(k1) == hash(k2)

    def test_set_dedup(self):
        s = {ExpertKey(1, 0), ExpertKey(1, 0), ExpertKey(2, 0)}
        assert len(s) == 2

    def test_equality(self):
        assert ExpertKey(1, 5) == ExpertKey(1, 5)
        assert ExpertKey(1, 5) != ExpertKey(1, 6)
        assert ExpertKey(1, 5) != ExpertKey(2, 5)

    def test_ordering(self):
        keys = [ExpertKey(2, 0), ExpertKey(1, 5), ExpertKey(1, 3)]
        assert sorted(keys) == [ExpertKey(1, 3), ExpertKey(1, 5), ExpertKey(2, 0)]

    def test_tuple_unpacking(self):
        layer, expert = ExpertKey(10, 200)
        assert layer == 10
        assert expert == 200


class TestEnumInterop:
    def test_cache_zone_matches_shm_protocol(self):
        assert CacheZone.STABLE == ZONE_STABLE
        assert CacheZone.STREAMING == ZONE_STREAMING

    def test_gpu_tier_matches(self):
        assert GpuTier.ABSENT == GPU_TIER_ABSENT
        assert GpuTier.HOT == GPU_TIER_HOT
        assert GpuTier.TRANSFERRING == GPU_TIER_TRANSFERRING

    def test_host_tier_matches(self):
        assert HostTier.COLD == HOST_TIER_COLD
        assert HostTier.LOADING_TO_RAM == HOST_TIER_LOADING
        assert HostTier.WARM == HOST_TIER_WARM

    def test_int_comparison(self):
        assert GpuTier.HOT > GpuTier.ABSENT
        assert HostTier.WARM > HostTier.COLD
        assert GpuTier.HOT == 6
        assert HostTier.WARM == 2


class TestGpuConfig:
    def test_construction(self):
        g = GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3)
        assert g.position == 0
        assert g.gpu_type == "rtx5090"
        assert g.is_tp is True

    def test_frozen(self):
        g = GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3)
        with pytest.raises(AttributeError):
            g.position = 1


class TestEngineMetadata:
    def test_construction(self, mock_engine_metadata):
        m = mock_engine_metadata
        assert m.num_gpus == 4
        assert m.num_moe_layers == 58
        assert m.num_experts == 256
        assert len(m.gpus) == 4
        assert m.gpus[0].is_tp is True
        assert m.gpus[2].is_tp is False

    def test_frozen(self, mock_engine_metadata):
        with pytest.raises(AttributeError):
            mock_engine_metadata.num_gpus = 8

    def test_default_empty_gpus(self):
        m = EngineMetadata(
            num_gpus=1, num_moe_layers=1, num_experts=1,
            num_layers=1, expert_bytes=100, kv_bytes_per_page=100,
        )
        assert m.gpus == ()


class TestPrefetchHint:
    def test_construction(self):
        h = PrefetchHint(
            key=ExpertKey(3, 10),
            target_layer=3,
            confidence=PrefetchConfidence.HIGH,
            source=PrefetchSource.PRESCOPE,
        )
        assert h.key == ExpertKey(3, 10)
        assert h.score == 0.0

    def test_score_default(self):
        h = PrefetchHint(
            key=ExpertKey(1, 0), target_layer=1,
            confidence=PrefetchConfidence.MEDIUM,
            source=PrefetchSource.PROBE, score=0.85,
        )
        assert h.score == 0.85


class TestAffinityHint:
    def test_construction(self):
        a = AffinityHint(key=ExpertKey(5, 100), preferred_gpu=2, score=0.95)
        assert a.key.layer_idx == 5
        assert a.preferred_gpu == 2

    def test_frozen(self):
        a = AffinityHint(key=ExpertKey(5, 100), preferred_gpu=2, score=0.95)
        with pytest.raises(AttributeError):
            a.score = 0.5


class TestWorkItemExperts:
    def test_variable_expert_count(self):
        w = WorkItemExperts(layer_idx=10)
        assert w.expert_keys == []
        w.expert_keys.append(ExpertKey(10, 5))
        w.expert_keys.append(ExpertKey(10, 42))
        assert len(w.expert_keys) == 2

    def test_with_routing_weights(self):
        w = WorkItemExperts(
            layer_idx=3,
            expert_keys=[ExpertKey(3, 0), ExpertKey(3, 1), ExpertKey(3, 2)],
            routing_weights=[0.5, 0.3, 0.2],
        )
        assert len(w.expert_keys) == len(w.routing_weights)
        assert sum(w.routing_weights) == pytest.approx(1.0)
