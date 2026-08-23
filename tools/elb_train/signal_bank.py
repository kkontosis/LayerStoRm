"""P11 Stage-0 free-signal bank: zero-training expert predictors
(spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(3)).

Every member follows a strictly ONLINE-CAUSAL protocol: at committed
position t it must predict BEFORE it sees t's routing (predict() then
update()). Global tables (token memo, transition EMA) persist across
sequences — supervision is a byproduct of serving and sequences are
independent, so cross-sequence accumulation is causal; per-sequence
state (b0_prev, EMAs, trailing window) resets at sequence start.

Members (state is [J, E] vectorized over layers; scores are arbitrary
monotone units — consumers only rank them):

  b0_prev       1 for last position's top-8 (the baseline, control)
  score_ema     EMA of the full selection-score vector g(t)
  trail16_freq  count of routes in the trailing 16 positions
  token_memo    SiDA-class per-(layer, token-bucket) expert counts;
                the only member whose prediction depends on the INPUT
                token (known at manifest time — it is the draft block)
  trans_ema     MoE-Infinity-class expert-transition counts:
                s_e = sum_{p in top8(t-1)} T_j[p, e], lazy-decayed
  xlayer        cross-layer score coherence: layer j predicted by layer
                j - delta's scores AT THE SAME POSITION. NOT a
                history predictor (uses the current forward's earlier
                layers) — the offline SCORE-SPACE ANALOG of the
                stale-router arm W_g^{j+delta} h^j, which is
                corpus-infeasible (no target-side hiddens were
                captured). Model-side diagnostic only; excluded from
                manifest/eviction proxies. Handled by the driver, not
                a BankMember (different causality).

Deployed-cost stamps (cost_us on the round critical path, state_bytes)
are metadata for the report — the budget class is <= 1 GEMV/layer/row
inline, <= 100 us/round host-side (§3-P11(1))."""

from __future__ import annotations

import collections

import numpy as np

TOKEN_BUCKETS = 4096


class BankMember:
    """Base: per-layer score state. predict(token) -> [J, E] float32."""

    name = "base"
    cost_us = 0.0
    state_bytes = 0
    warmable = False        # True: global tables benefit from a train pass
    sparse = True           # count-type scores: 0 == "predicts nothing";
    #                         consumers must not rank zero-score experts
    #                         (dense members like score EMAs set False)

    def __init__(self, n_layers: int, n_experts: int):
        self.J = n_layers
        self.E = n_experts

    def reset_seq(self) -> None:
        """Clear per-sequence state (never the global tables)."""

    def predict(self, token: int) -> np.ndarray:
        raise NotImplementedError

    def update(self, tops: np.ndarray, top_w: np.ndarray,
               sel: np.ndarray | None, token: int) -> None:
        """tops [J, K] int32, top_w [J, K] f32, sel [J, E] f32 selection
        scores (None when the caller skipped logits), token: input token
        id AT this position."""
        raise NotImplementedError


class B0Prev(BankMember):
    name = "b0_prev"
    cost_us = 0.1
    state_bytes = 75 * 8 * 4

    def __init__(self, n_layers, n_experts):
        super().__init__(n_layers, n_experts)
        self.scores = np.zeros((self.J, self.E), np.float32)

    def reset_seq(self):
        self.scores[:] = 0.0

    def predict(self, token):
        return self.scores

    def update(self, tops, top_w, sel, token):
        self.scores[:] = 0.0
        np.put_along_axis(self.scores, tops.astype(np.int64), 1.0, axis=1)


class ScoreEma(BankMember):
    """EMA of the dense selection-score vector; decay = 0.5**(1/half_life).
    Engine form: one [J, E] f32 FMA per committed position."""
    cost_us = 1.0
    state_bytes = 75 * 256 * 4
    warmable = False
    sparse = False          # dense sel-space scores (bias may be negative)

    def __init__(self, n_layers, n_experts, half_life: float):
        super().__init__(n_layers, n_experts)
        self.name = f"score_ema{half_life:g}"
        self.decay = 0.5 ** (1.0 / half_life)
        self.state = np.zeros((self.J, self.E), np.float32)
        self.norm = 0.0                     # bias-corrected EMA

    def reset_seq(self):
        self.state[:] = 0.0
        self.norm = 0.0

    def predict(self, token):
        if self.norm == 0.0:
            return self.state
        return self.state / self.norm

    def update(self, tops, top_w, sel, token):
        self.state *= self.decay
        self.state += (1.0 - self.decay) * sel
        self.norm = self.norm * self.decay + (1.0 - self.decay)


class TrailFreq(BankMember):
    """Route counts over the trailing `horizon` committed positions."""
    cost_us = 1.0
    state_bytes = 75 * 256 * 2 + 16 * 75 * 8 * 4

    def __init__(self, n_layers, n_experts, horizon: int = 16):
        super().__init__(n_layers, n_experts)
        self.name = f"trail{horizon}_freq"
        self.horizon = horizon
        self.counts = np.zeros((self.J, self.E), np.float32)
        self.ring: collections.deque = collections.deque()

    def reset_seq(self):
        self.counts[:] = 0.0
        self.ring.clear()

    def predict(self, token):
        return self.counts

    def update(self, tops, top_w, sel, token):
        t64 = tops.astype(np.int64)
        if len(self.ring) >= self.horizon:
            old = self.ring.popleft()
            np.add.at(self.counts, (np.arange(self.J)[:, None], old), -1.0)
        self.ring.append(t64)
        np.add.at(self.counts, (np.arange(self.J)[:, None], t64), 1.0)


class TokenMemo(BankMember):
    """Per-(layer, token-bucket) expert counts (SiDA-MoE-class data-aware
    memo). Offline: dense int32 [J, BUCKETS, E] (~300 MB); the deployed
    form is a per-(j, bucket) top-M sparse dictionary."""
    name = "token_memo"
    cost_us = 2.0
    state_bytes = 75 * TOKEN_BUCKETS * 16 * 6   # deployed sparse top-16
    warmable = True

    def __init__(self, n_layers, n_experts, buckets: int = TOKEN_BUCKETS):
        super().__init__(n_layers, n_experts)
        self.buckets = buckets
        self.table = np.zeros((n_layers, buckets, n_experts), np.int32)

    def bucket(self, token: int) -> int:
        return int(token) % self.buckets

    def predict(self, token):
        return self.table[:, self.bucket(token), :].astype(np.float32)

    def update(self, tops, top_w, sel, token):
        b = self.bucket(token)
        np.add.at(self.table, (np.arange(self.J)[:, None], b,
                               tops.astype(np.int64)), 1)


class TransEma(BankMember):
    """Expert-transition EMA (MoE-Infinity-class): T_j[p, e] ~ how often
    expert e is routed at t given p was routed at t-1. Lazy global decay
    via a scale factor (renormalized before underflow); prediction
    s_e = sum_{p in top8(t-1)} T_j[p, e]."""
    name = "trans_ema"
    cost_us = 2.0
    state_bytes = 75 * 256 * 16 * 6             # deployed sparse top-16
    warmable = True

    def __init__(self, n_layers, n_experts, decay: float = 0.99):
        super().__init__(n_layers, n_experts)
        self.decay = decay
        self.table = np.zeros((n_layers, n_experts, n_experts), np.float32)
        self.scale = 1.0
        self.prev: np.ndarray | None = None     # [J, K] int64, per-seq

    def reset_seq(self):
        self.prev = None

    def predict(self, token):
        if self.prev is None:
            return np.zeros((self.J, self.E), np.float32)
        rows = self.table[np.arange(self.J)[:, None], self.prev]  # [J,K,E]
        return rows.sum(axis=1)                 # scale cancels in ranking

    def update(self, tops, top_w, sel, token):
        t64 = tops.astype(np.int64)
        if self.prev is not None:
            self.scale *= self.decay
            w = 1.0 / self.scale
            j_idx = np.arange(self.J)[:, None, None]
            self.table[j_idx, self.prev[:, :, None], t64[:, None, :]] += w
            if self.scale < 1e-20:
                self.table *= self.scale
                self.scale = 1.0
        self.prev = t64


def build_bank(n_layers: int, n_experts: int, *, warmed: bool
               ) -> list[BankMember]:
    """The benched members. `warmed` only labels the stateful copies —
    the driver runs the train pass on one bank and not the other."""
    suffix = "_warm" if warmed else "_cold"
    members: list[BankMember] = []
    if not warmed:
        # per-seq-state members are split-agnostic; bench them once.
        members += [B0Prev(n_layers, n_experts),
                    ScoreEma(n_layers, n_experts, 2.0),
                    ScoreEma(n_layers, n_experts, 8.0),
                    TrailFreq(n_layers, n_experts, 16)]
    memo = TokenMemo(n_layers, n_experts)
    memo.name += suffix
    trans = TransEma(n_layers, n_experts)
    trans.name += suffix
    members += [memo, trans]
    return members
