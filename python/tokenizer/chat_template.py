"""Jinja2 chat template renderer for OpenAI-format messages.

Loads a model's chat template from a standalone .jinja file, an embedded
field in tokenizer_config.json, or falls back to a built-in ChatML default.
Renders messages to a model-native prompt string in a sandboxed Jinja
environment. No HuggingFace dependency — only jinja2 + stdlib.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
import jinja2.sandbox

from tokenizer.tokenizer_wrapper import _read_json

log = logging.getLogger(__name__)

_DEFAULT_CHATML_TEMPLATE = """\
{% for message in messages %}\
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}\
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\
<|im_start|>assistant
{% endif %}\
"""


class ChatTemplateError(Exception):
    """Raised when chat template rendering fails."""


def _raise_exception(msg: str) -> None:
    raise ChatTemplateError(msg)


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def _tojson(value: Any, indent: int | None = None, **kwargs: Any) -> str:
    # Templates may pass ensure_ascii explicitly (GLM-5.2 does
    # `tojson(ensure_ascii=False)`) — don't collide with our default.
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(value, indent=indent, **kwargs)


def _from_json(value: Any) -> Any:
    # HF transformers exposes a `from_json` filter to chat templates
    # (DeepSeek-V4's template uses it on string tool arguments) — mirror
    # it; non-string inputs pass through.
    if isinstance(value, str):
        return json.loads(value)
    return value


def _resolve_template(model_dir: Path) -> tuple[str, str]:
    jinja_file = model_dir / "chat_template.jinja"
    if jinja_file.is_file():
        try:
            text = jinja_file.read_text(encoding="utf-8")
            return text, f"file:{jinja_file.name}"
        except OSError:
            pass

    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    if tokenizer_config is not None:
        raw = tokenizer_config.get("chat_template")
        if isinstance(raw, str) and raw.strip():
            return raw, "tokenizer_config.json"
        if isinstance(raw, dict):
            tmpl = raw.get("default") or next(iter(raw.values()), None)
            if isinstance(tmpl, str) and tmpl.strip():
                return tmpl, "tokenizer_config.json"

    return _DEFAULT_CHATML_TEMPLATE, "default:ChatML"


def _extract_token_str(tokenizer_config: dict | None, key: str) -> str:
    if tokenizer_config is None:
        return ""
    val = tokenizer_config.get(key)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("content", "")
    return ""


def _normalize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            msg["content"] = "".join(parts) if parts else ""

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if fn is None:
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        pass
    return out


class ChatTemplateRenderer:
    """Renders OpenAI-format messages into model-native prompt strings."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        template_override: str | None = None,
    ) -> None:
        model_dir = Path(model_path)

        if template_override is not None:
            template_str = template_override
            self._source = "override"
        elif model_dir.is_dir():
            template_str, self._source = _resolve_template(model_dir)
        else:
            log.warning("model_path %s is not a directory, using ChatML", model_path)
            template_str = _DEFAULT_CHATML_TEMPLATE
            self._source = "default:ChatML"

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        env.filters["tojson"] = _tojson
        env.filters["from_json"] = _from_json

        try:
            self._template = env.from_string(template_str)
        except jinja2.TemplateSyntaxError as exc:
            raise ChatTemplateError(
                f"invalid template syntax: {exc}",
            ) from exc

        tokenizer_config = (
            _read_json(model_dir / "tokenizer_config.json")
            if model_dir.is_dir() else None
        )
        self._bos_token = _extract_token_str(tokenizer_config, "bos_token")
        self._eos_token = _extract_token_str(tokenizer_config, "eos_token")

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        thinking: bool | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        **extra_kwargs: Any,
    ) -> str:
        """Render messages to a prompt string.

        ``enable_thinking`` / ``reasoning_effort`` are the GLM-5 template's
        native knobs (GLM defaults when omitted: thinking ON, effort Max —
        the template treats any effort other than "high" as "max"). They are
        injected only when set, so templates that don't use them are
        unaffected. ``thinking`` (other templates' name for the same switch)
        doubles as ``enable_thinking`` when the latter isn't given.
        """
        normalized = _normalize_messages(messages)
        context: dict[str, Any] = {
            "messages": normalized,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self._bos_token,
            "eos_token": self._eos_token,
        }
        if tools is not None:
            context["tools"] = tools
        if thinking is not None:
            context["thinking"] = thinking
            if enable_thinking is None:
                enable_thinking = thinking
        if enable_thinking is not None:
            context["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            context["reasoning_effort"] = reasoning_effort
        context.update(extra_kwargs)
        try:
            return self._template.render(**context)
        except ChatTemplateError:
            raise
        except jinja2.TemplateError as exc:
            raise ChatTemplateError(
                f"template rendering failed: {exc}",
            ) from exc

    @property
    def template_source(self) -> str:
        return self._source
