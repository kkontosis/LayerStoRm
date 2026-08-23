"""Tests for ConfigHotReload and CMD_CONFIG_UPDATE integration."""

import struct

import pytest

from orchestrator.config_hot_reload import ConfigHotReload
from orchestrator.command_writer import CommandWriter
from orchestrator.shm_protocol import CMD_CONFIG_UPDATE


class TestConfigHotReload:
    def test_init_has_changeable_fields(self):
        hr = ConfigHotReload()
        paths = hr.changeable_paths
        assert len(paths) > 0
        assert "prefetch.prescope.top_k" in paths
        assert "speculation.enabled" in paths

    def test_init_sets_defaults(self):
        hr = ConfigHotReload()
        assert hr.prefetch_prescope_top_k == 8
        assert hr.prefetch_prescope_score_threshold == pytest.approx(0.01)
        assert hr.speculation_enabled is True
        assert hr.prefetch_prescope_predictor_enabled is False
        assert hr._internal_prescope_learning_rate == pytest.approx(0.001)

    def test_update_changes_value(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.top_k", 16)
        assert hr.prefetch_prescope_top_k == 16
        assert hr.get("prefetch.prescope.top_k") == 16

    def test_update_marks_dirty(self):
        hr = ConfigHotReload()
        assert not hr.has_pending
        hr.update("prefetch.prescope.top_k", 16)
        assert hr.has_pending

    def test_clear_pending(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.top_k", 16)
        hr.clear_pending()
        assert not hr.has_pending

    def test_update_unknown_field_raises(self):
        hr = ConfigHotReload()
        with pytest.raises(KeyError, match="not a changeable"):
            hr.update("model.hidden_size", 4096)

    def test_get_unknown_field_raises(self):
        hr = ConfigHotReload()
        with pytest.raises(KeyError, match="not a changeable"):
            hr.get("model.hidden_size")

    def test_pending_daemon_updates_bool(self):
        hr = ConfigHotReload()
        hr.update("speculation.enabled", False)
        updates = hr.pending_daemon_updates()
        assert len(updates) == 1
        fid, vtype, raw = updates[0]
        assert vtype == 0  # bool
        assert raw == 0  # False

    def test_pending_daemon_updates_int(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.top_k", 12)
        updates = hr.pending_daemon_updates()
        assert len(updates) == 1
        fid, vtype, raw = updates[0]
        assert vtype == 1  # int
        assert raw == 12

    def test_pending_daemon_updates_float(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.score_threshold", 0.05)
        updates = hr.pending_daemon_updates()
        assert len(updates) == 1
        fid, vtype, raw = updates[0]
        assert vtype == 2  # float
        expected_raw = struct.unpack("<I", struct.pack("<f", 0.05))[0]
        assert raw == expected_raw

    def test_multiple_updates(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.top_k", 16)
        hr.update("speculation.enabled", False)
        hr.update("prefetch.prescope.score_threshold", 0.05)
        updates = hr.pending_daemon_updates()
        assert len(updates) == 3

    def test_update_same_field_twice(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.top_k", 16)
        hr.update("prefetch.prescope.top_k", 24)
        assert hr.prefetch_prescope_top_k == 24
        updates = hr.pending_daemon_updates()
        assert len(updates) == 1
        assert updates[0][2] == 24


class TestCommandWriterConfigUpdate:
    def test_config_update_command(self):
        writer = CommandWriter(initial_seq=100)
        updates = [(114, 1, 16), (131, 0, 0)]
        cmd = writer.config_update(updates)
        assert cmd.cmd_type == CMD_CONFIG_UPDATE
        assert cmd.cmd_seq == 100
        assert cmd.payload.config_update.count == 2
        assert cmd.payload.config_update.entries[0].field_id == 114
        assert cmd.payload.config_update.entries[0].value_type == 1
        assert cmd.payload.config_update.entries[0].raw_value == 16
        assert cmd.payload.config_update.entries[1].field_id == 131
        assert cmd.payload.config_update.entries[1].value_type == 0
        assert cmd.payload.config_update.entries[1].raw_value == 0

    def test_config_update_too_many_raises(self):
        writer = CommandWriter(initial_seq=0)
        updates = [(i, 1, i) for i in range(30)]
        with pytest.raises(ValueError, match="Too many"):
            writer.config_update(updates)

    def test_round_trip_hot_reload_to_command(self):
        hr = ConfigHotReload()
        hr.update("prefetch.prescope.top_k", 32)
        hr.update("_internal-prescope.learning_rate", 0.01)
        updates = hr.pending_daemon_updates()
        writer = CommandWriter(initial_seq=0)
        cmd = writer.config_update(updates)
        assert cmd.cmd_type == CMD_CONFIG_UPDATE
        assert cmd.payload.config_update.count == 2
        hr.clear_pending()
        assert not hr.has_pending
