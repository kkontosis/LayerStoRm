"""Tests for chat template renderer."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import pytest

from tokenizer.chat_template import (
    ChatTemplateError,
    ChatTemplateRenderer,
    _extract_token_str,
    _normalize_messages,
    _resolve_template,
)

_TEST_DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "test-data"
_KIMI = _TEST_DATA / "Kimi-K2.5"
_DEEPSEEK = _TEST_DATA / "DeepSeek-V3.2"
_GLM5 = _TEST_DATA / "GLM-5"


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

class TestTemplateResolution:

    def test_jinja_file_found(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text(
            "{{ messages[0]['content'] }}",
        )
        tmpl, source = _resolve_template(tmp_path)
        assert "messages[0]" in tmpl
        assert source == "file:chat_template.jinja"

    def test_embedded_in_tokenizer_config(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "EMBEDDED:{{ messages|length }}",
        }))
        tmpl, source = _resolve_template(tmp_path)
        assert tmpl.startswith("EMBEDDED:")
        assert source == "tokenizer_config.json"

    def test_embedded_dict_format(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": {
                "default": "DEFAULT:{{ messages|length }}",
                "tool_use": "TOOL:{{ messages|length }}",
            },
        }))
        tmpl, source = _resolve_template(tmp_path)
        assert tmpl.startswith("DEFAULT:")
        assert source == "tokenizer_config.json"

    def test_embedded_dict_no_default_uses_first(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": {
                "tool_use": "TOOL:{{ messages|length }}",
            },
        }))
        tmpl, source = _resolve_template(tmp_path)
        assert tmpl.startswith("TOOL:")

    def test_fallback_to_chatml(self, tmp_path):
        tmpl, source = _resolve_template(tmp_path)
        assert "<|im_start|>" in tmpl
        assert source == "default:ChatML"

    def test_jinja_file_takes_priority(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text("FILE_TEMPLATE")
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "EMBEDDED_TEMPLATE",
        }))
        tmpl, _ = _resolve_template(tmp_path)
        assert tmpl == "FILE_TEMPLATE"

    def test_kimi_real_template_loads(self):
        tmpl, source = _resolve_template(_KIMI)
        assert source == "file:chat_template.jinja"
        assert "<|im_user|>" in tmpl
        assert "<|tool_calls_section_begin|>" in tmpl


# ---------------------------------------------------------------------------
# ChatML default rendering
# ---------------------------------------------------------------------------

class TestChatMLDefault:

    def _renderer(self, tmp_path):
        return ChatTemplateRenderer(tmp_path)

    def test_simple_conversation(self, tmp_path):
        r = self._renderer(tmp_path)
        result = r.render([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ])
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHi<|im_end|>" in result
        assert result.rstrip().endswith("<|im_start|>assistant")

    def test_add_generation_prompt_false(self, tmp_path):
        r = self._renderer(tmp_path)
        result = r.render(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=False,
        )
        assert "<|im_start|>assistant" not in result

    def test_user_only(self, tmp_path):
        r = self._renderer(tmp_path)
        result = r.render([{"role": "user", "content": "Hello"}])
        assert "<|im_start|>user\nHello<|im_end|>" in result

    def test_template_source_is_chatml(self, tmp_path):
        r = self._renderer(tmp_path)
        assert r.template_source == "default:ChatML"

    def test_template_override(self, tmp_path):
        r = ChatTemplateRenderer(
            tmp_path,
            template_override="CUSTOM:{{ messages|length }}",
        )
        assert r.render([{"role": "user", "content": "x"}]) == "CUSTOM:1"
        assert r.template_source == "override"


# ---------------------------------------------------------------------------
# Kimi K2.5 real template
# ---------------------------------------------------------------------------

class TestKimiK25Template:

    @pytest.fixture
    def renderer(self):
        return ChatTemplateRenderer(_KIMI)

    def test_basic_chat(self, renderer):
        result = renderer.render([
            {"role": "user", "content": "Hello"},
        ])
        assert "<|im_user|>" in result
        assert "<|im_middle|>" in result
        assert "Hello" in result
        assert "<|im_end|>" in result

    def test_generation_prompt_with_think(self, renderer):
        result = renderer.render([
            {"role": "user", "content": "Hi"},
        ])
        assert "<|im_assistant|>assistant<|im_middle|>" in result
        assert "<think>" in result

    def test_thinking_false(self, renderer):
        result = renderer.render(
            [{"role": "user", "content": "Hi"}],
            thinking=False,
        )
        assert "<think></think>" in result

    def test_with_tools(self, renderer):
        tools = [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object"},
        }}]
        result = renderer.render(
            [{"role": "user", "content": "Weather?"}],
            tools=tools,
        )
        assert "<|im_system|>tool_declare<|im_middle|>" in result
        assert "get_weather" in result

    def test_with_tool_calls(self, renderer):
        result = renderer.render(
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        },
                    }],
                },
                {
                    "role": "tool",
                    "content": "Sunny, 72F",
                    "tool_call_id": "call_1",
                },
            ],
            add_generation_prompt=True,
        )
        assert "<|tool_calls_section_begin|>" in result
        assert "<|tool_call_begin|>" in result
        assert "call_1" in result

    def test_source_is_file(self, renderer):
        assert renderer.template_source == "file:chat_template.jinja"


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------

class TestMessageNormalization:

    def test_string_content_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _normalize_messages(msgs)
        assert result[0]["content"] == "hello"

    def test_multimodal_text_extracted(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Look at "},
            {"type": "image_url", "image_url": {"url": "http://..."}},
            {"type": "text", "text": "this image"},
        ]}]
        result = _normalize_messages(msgs)
        assert result[0]["content"] == "Look at this image"

    def test_none_content_unchanged(self):
        msgs = [{"role": "assistant", "content": None}]
        result = _normalize_messages(msgs)
        assert result[0]["content"] is None

    def test_tool_call_arguments_parsed(self):
        msgs = [{"role": "assistant", "tool_calls": [{
            "id": "1",
            "function": {"name": "f", "arguments": '{"a": 1}'},
        }]}]
        result = _normalize_messages(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_tool_call_arguments_already_dict(self):
        msgs = [{"role": "assistant", "tool_calls": [{
            "id": "1",
            "function": {"name": "f", "arguments": {"a": 1}},
        }]}]
        result = _normalize_messages(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_original_messages_not_mutated(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
        ]}]
        _normalize_messages(msgs)
        assert isinstance(msgs[0]["content"], list)

    def test_multimodal_no_text_parts(self):
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://..."}},
        ]}]
        result = _normalize_messages(msgs)
        assert result[0]["content"] == ""


# ---------------------------------------------------------------------------
# BOS/EOS token extraction
# ---------------------------------------------------------------------------

class TestBosEosExtraction:

    def test_plain_string_kimi(self):
        config = {"bos_token": "[BOS]", "eos_token": "[EOS]"}
        assert _extract_token_str(config, "bos_token") == "[BOS]"
        assert _extract_token_str(config, "eos_token") == "[EOS]"

    def test_added_token_dict_deepseek(self):
        config = {
            "bos_token": {
                "__type": "AddedToken",
                "content": "<｜begin▁of▁sentence｜>",
            },
        }
        assert _extract_token_str(config, "bos_token") == "<｜begin▁of▁sentence｜>"

    def test_missing_key(self):
        assert _extract_token_str({}, "bos_token") == ""

    def test_none_config(self):
        assert _extract_token_str(None, "bos_token") == ""

    def test_real_deepseek_config(self):
        import json as _json
        cfg = _json.loads((_DEEPSEEK / "tokenizer_config.json").read_text())
        bos = _extract_token_str(cfg, "bos_token")
        eos = _extract_token_str(cfg, "eos_token")
        assert "begin" in bos
        assert "end" in eos

    def test_real_kimi_config(self):
        import json as _json
        cfg = _json.loads((_KIMI / "tokenizer_config.json").read_text())
        assert _extract_token_str(cfg, "bos_token") == "[BOS]"
        assert _extract_token_str(cfg, "eos_token") == "[EOS]"


# ---------------------------------------------------------------------------
# Jinja environment
# ---------------------------------------------------------------------------

class TestJinjaEnvironment:

    def test_raise_exception_works(self, tmp_path):
        r = ChatTemplateRenderer(
            tmp_path,
            template_override="{{ raise_exception('test error') }}",
        )
        with pytest.raises(ChatTemplateError, match="test error"):
            r.render([{"role": "user", "content": "x"}])

    def test_strftime_now_returns_year(self, tmp_path):
        r = ChatTemplateRenderer(
            tmp_path,
            template_override="{{ strftime_now('%Y') }}",
        )
        result = r.render([])
        assert result == str(datetime.now().year)

    def test_loopcontrols_break(self, tmp_path):
        tmpl = (
            "{% for m in messages %}"
            "{{ m['role'] }}"
            "{% if loop.first %}{% break %}{% endif %}"
            "{% endfor %}"
        )
        r = ChatTemplateRenderer(tmp_path, template_override=tmpl)
        result = r.render([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        assert result == "user"

    def test_tojson_filter(self, tmp_path):
        tmpl = "{{ tools | tojson }}"
        r = ChatTemplateRenderer(tmp_path, template_override=tmpl)
        result = r.render([], tools=[{"name": "f"}])
        assert '"name": "f"' in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_invalid_template_syntax(self, tmp_path):
        with pytest.raises(ChatTemplateError, match="invalid template syntax"):
            ChatTemplateRenderer(
                tmp_path,
                template_override="{% if %}",
            )

    def test_missing_model_path_uses_chatml(self, tmp_path):
        r = ChatTemplateRenderer(tmp_path / "nonexistent")
        assert r.template_source == "default:ChatML"

    def test_rendering_error_wrapped(self, tmp_path):
        r = ChatTemplateRenderer(
            tmp_path,
            template_override="{{ undefined_var.method() }}",
        )
        with pytest.raises(ChatTemplateError, match="rendering failed"):
            r.render([])

    def test_extra_kwargs_passed_to_template(self, tmp_path):
        r = ChatTemplateRenderer(
            tmp_path,
            template_override="{{ custom_var }}",
        )
        result = r.render([], custom_var="hello")
        assert result == "hello"


# ---------------------------------------------------------------------------
# GLM-5.2 (glm_moe_dsa) — reasoning_effort / enable_thinking / tool calls
# ---------------------------------------------------------------------------

_GLM52 = _TEST_DATA / "GLM-5.2"


@pytest.mark.skipif(not _GLM52.is_dir(), reason="GLM-5.2 test data absent")
class TestGlm52Template:

    @pytest.fixture()
    def renderer(self):
        return ChatTemplateRenderer(_GLM52)

    _MSGS = [{"role": "user", "content": "hi"}]

    def test_default_prefix_and_effort_max(self, renderer):
        out = renderer.render(self._MSGS)
        assert out.startswith("[gMASK]<sop>")
        # GLM defaults: thinking ON, effort Max (any value but "high" → max)
        assert "<|system|>Reasoning Effort: Max" in out
        assert out.rstrip().endswith("<think>")

    def test_reasoning_effort_high(self, renderer):
        out = renderer.render(self._MSGS, reasoning_effort="high")
        assert "<|system|>Reasoning Effort: High" in out

    def test_reasoning_effort_other_values_map_to_max(self, renderer):
        out = renderer.render(self._MSGS, reasoning_effort="low")
        assert "<|system|>Reasoning Effort: Max" in out

    def test_enable_thinking_false(self, renderer):
        out = renderer.render(self._MSGS, enable_thinking=False)
        assert "Reasoning Effort" not in out
        assert out.rstrip().endswith("<think></think>")

    def test_thinking_alias_maps_to_enable_thinking(self, renderer):
        out = renderer.render(self._MSGS, thinking=False)
        assert "Reasoning Effort" not in out
        assert out.rstrip().endswith("<think></think>")

    def test_tool_call_render_roundtrip(self, renderer):
        from tokenizer.tool_call_parser import get_tool_call_parser
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps(
                        {"city": "Paris", "days": 3, "metric": True}),
                },
            }]},
        ]
        out = renderer.render(msgs, add_generation_prompt=False)
        assert "<tool_call>get_weather" in out
        assert "<arg_key>city</arg_key><arg_value>Paris</arg_value>" in out
        # Parse the rendered wire format back (same format the model emits).
        seg = out[out.index("<tool_call>"):]
        seg = seg[: seg.index("</tool_call>") + len("</tool_call>")]
        parsed = get_tool_call_parser("glm_moe_dsa").parse(seg)
        assert len(parsed.tool_calls) == 1
        tc = parsed.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args == {"city": "Paris", "days": 3, "metric": True}
