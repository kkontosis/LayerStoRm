"""Prompt lookup — zero-cost N-gram suffix matching for draft token generation.

Scans the existing token sequence (prompt + prior generation) for N-gram
suffixes matching the tail, then proposes continuation tokens from the matched
location. First priority in the speculation pipeline (INV-0.8d): checked
before MTP and self-speculative drafts because it has zero GPU compute cost.

Algorithm adapted from vLLM ngram_proposer.py (Apache-2.0).
Reference: "Prompt Lookup Decoding" (Saxena).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLookupConfig:
    enabled: bool = True
    min_ngram_size: int = 2
    max_ngram_size: int = 4
    max_continuation_length: int = 5
    acceptance_ema_alpha: float = 0.3


class PromptLookup:

    def __init__(self, config: PromptLookupConfig | None = None) -> None:
        self._config = config or PromptLookupConfig()
        self._total_lookups: int = 0
        self._total_matches: int = 0
        self._total_proposed: int = 0
        self._total_accepted: int = 0
        self._ema_acceptance_rate: float = 0.0
        self._ema_initialized: bool = False

    def lookup(self, token_ids: list[int],
               max_continuation: int | None = None) -> list[int]:
        self._total_lookups += 1

        if not self._config.enabled:
            return []

        seq_len = len(token_ids)
        if seq_len < self._config.min_ngram_size + 1:
            return []

        max_cont = (max_continuation if max_continuation is not None
                     else self._config.max_continuation_length)
        result = self._find_match(token_ids, seq_len, max_cont)

        if result:
            self._total_matches += 1

        return result

    def _find_match(self, token_ids: list[int], seq_len: int,
                    max_cont: int) -> list[int]:
        max_n = min(self._config.max_ngram_size, seq_len - 1)
        min_n = self._config.min_ngram_size

        for n in range(max_n, min_n - 1, -1):
            suffix_start = seq_len - n

            for pos in range(suffix_start - 1, -1, -1):
                matched = True
                for j in range(n):
                    if token_ids[pos + j] != token_ids[suffix_start + j]:
                        matched = False
                        break
                if matched:
                    cont_start = pos + n
                    cont_end = min(cont_start + max_cont, seq_len)
                    if cont_start < cont_end:
                        return token_ids[cont_start:cont_end]

        return []

    def record_result(self, num_proposed: int, num_accepted: int) -> None:
        self._total_proposed += num_proposed
        self._total_accepted += num_accepted

        if num_proposed > 0:
            batch_rate = num_accepted / num_proposed
            if not self._ema_initialized:
                self._ema_acceptance_rate = batch_rate
                self._ema_initialized = True
            else:
                a = self._config.acceptance_ema_alpha
                self._ema_acceptance_rate = (
                    a * batch_rate + (1.0 - a) * self._ema_acceptance_rate
                )

    @property
    def acceptance_rate(self) -> float:
        return self._ema_acceptance_rate

    @property
    def match_rate(self) -> float:
        if self._total_lookups == 0:
            return 0.0
        return self._total_matches / self._total_lookups

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def total_lookups(self) -> int:
        return self._total_lookups

    @property
    def total_matches(self) -> int:
        return self._total_matches

    @property
    def total_proposed(self) -> int:
        return self._total_proposed

    @property
    def total_accepted(self) -> int:
        return self._total_accepted
