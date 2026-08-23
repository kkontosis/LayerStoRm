"""Per-request context annotations — incrementally-maintained metadata.

Each active generation context has a ContextAnnotation holding trackers that
update on every generated token. Trackers implement ContextAnnotationTracker
(reset, update_token, update_batch, rebuild).

Current trackers:
  ReasoningModeTracker — detects <think>/<​/think> markers (INV-4.8)

Future tracker ideas (not implemented):
  - Repetition/loop detector (n-gram frequency spike)
  - Structured output bracket depth ({/}, [/])
  - Token count / generation length stats
  - Language detection (script histogram)

Registry: ContextAnnotationRegistry maintains dict[request_id, ContextAnnotation].
Standalone until #62 absorbs it into per-request state (TD-15).

Lifecycle: annotations live only for active generations. Dropped on request
completion. rebuild() handles cold-start from token history if needed.
Cross-session caching deferred (TD-16).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContextAnnotationTracker(Protocol):

    def reset(self) -> None: ...

    def update_token(self, token_id: int) -> None: ...

    def update_batch(self, token_ids: list[int]) -> None: ...

    def rebuild(self, full_token_ids: list[int]) -> None: ...


class ReasoningModeTracker:

    __slots__ = ("_start_id", "_end_id", "is_thinking")

    def __init__(self, think_start_token_id: int = -1,
                 think_end_token_id: int = -2) -> None:
        self._start_id = think_start_token_id
        self._end_id = think_end_token_id
        self.is_thinking: bool = False

    def reset(self) -> None:
        self.is_thinking = False

    def update_token(self, token_id: int) -> None:
        if token_id == self._start_id and self._start_id >= 0:
            self.is_thinking = True
        elif token_id == self._end_id and self._end_id >= 0:
            self.is_thinking = False

    def update_batch(self, token_ids: list[int]) -> None:
        for tid in token_ids:
            self.update_token(tid)

    def rebuild(self, full_token_ids: list[int]) -> None:
        self.reset()
        for tid in full_token_ids:
            self.update_token(tid)


class ContextAnnotation:

    __slots__ = ("reasoning",)

    def __init__(self, think_start_token_id: int = -1,
                 think_end_token_id: int = -2) -> None:
        self.reasoning = ReasoningModeTracker(
            think_start_token_id, think_end_token_id,
        )

    def _trackers(self) -> list[ContextAnnotationTracker]:
        return [self.reasoning]

    def reset(self) -> None:
        for t in self._trackers():
            t.reset()

    def update_token(self, token_id: int) -> None:
        for t in self._trackers():
            t.update_token(token_id)

    def update_batch(self, token_ids: list[int]) -> None:
        for t in self._trackers():
            t.update_batch(token_ids)

    def rebuild(self, full_token_ids: list[int]) -> None:
        for t in self._trackers():
            t.rebuild(full_token_ids)


class ContextAnnotationRegistry:

    __slots__ = ("_entries", "_start_id", "_end_id")

    def __init__(self, think_start_token_id: int = -1,
                 think_end_token_id: int = -2) -> None:
        self._entries: dict[int, ContextAnnotation] = {}
        self._start_id = think_start_token_id
        self._end_id = think_end_token_id

    def get_or_create(self, request_id: int) -> ContextAnnotation:
        if request_id not in self._entries:
            self._entries[request_id] = ContextAnnotation(
                self._start_id, self._end_id,
            )
        return self._entries[request_id]

    def get(self, request_id: int) -> ContextAnnotation | None:
        return self._entries.get(request_id)

    def remove(self, request_id: int) -> None:
        self._entries.pop(request_id, None)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, request_id: int) -> bool:
        return request_id in self._entries
