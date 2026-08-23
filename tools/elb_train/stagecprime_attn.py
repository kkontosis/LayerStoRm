"""P11.c' stage-3: the pre-registered §3-P11(4)(iii) tiny-attention
challenger, with the multi-token + expert-id-link directive.

A ~85k-parameter attention model over the trailing-W position window.
Each window slot contributes (token embedding-bucket embedding, routed
EXPERT-SET embedding — a learned [E, d] expert-id table, the direct
"link expert ids into the mix" arm — mean gate weight, draft conf,
slot-position embedding). Two modes:

  pooled        one query per layer (from a learned layer embedding);
                pooled context scored against a learned expert-output
                table -> 256 scores/layer.
  expert_query  one query PER (layer, expert): q = proj(exp_emb +
                layer_emb) attends over the window; scores from the
                attended vector — the richer expert-id link (x11).

Output is a RESIDUAL over the online ridge: challenger logits =
alpha * ridge_open_score (detached) + model_out; trained ONLINE with
per-position Adam steps on BCE against the revealed top-8 mask
(predict-then-update — metrics always use pre-update parameters).
Deployed-cost estimate (documented, not measured): pooled ~0.75M
MACs/round, expert_query ~8M MACs/round — µs-class on GPU, inside the
pre-registered ≤50 µs rent. Kill terms: must beat pool@32 0.7252 AND
novel-rec 0.586 on the same held protocol, else killed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_WINDOW = 8
D_EMB = 16
D_MODEL = 32
ATTN_LR = 1e-3

# ── x17 ctxformer (ARCHITECT_REVIEW.md Part B.6, user-requested
# direct experiment): ONE-head transformer over the trailing token
# window, output = ONE c' FEATURE column. Built under the x10-x13
# post-mortem constraints: identity-resolution FROZEN token features
# (no bucket coarsening), per-layer UNTIED expert tables (independent
# namespaces), score-space MSE on DECISION rows with a per-layer bias
# (no BCE link -> no calibration sink), ridge/Hedge floor via the
# feature composition. ~350k learned params (~1.4 MB f32 / 0.7 MB
# fp16); one forward per POSITION (not per expert) ~0.4M MACs ->
# µs-class, inside the 50 µs rent.
CTX_WINDOW = 8
CTX_D_TOK = 32          # frozen PCA token-feature dim
CTX_D = 32              # attention dim
CTX_D_OUT = 16          # per-layer context code dim
CTX_LR = 1e-3


def emb_project(emb: torch.Tensor, d: int = CTX_D_TOK,
                seed: int = 0) -> torch.Tensor:
    """Frozen token features: PCA of the model's embedding table to
    d dims, per-column standardized. Deterministic via seeded RNG."""
    x = emb.float()
    x = x - x.mean(dim=0, keepdim=True)
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    _, _, v = torch.pca_lowrank(x, q=d, niter=4)
    torch.random.set_rng_state(g)
    f = x @ v[:, :d]
    return (f / f.std(dim=0, keepdim=True).clamp(min=1e-6)).float()


class CtxEncoder(nn.Module):
    """query = current token; keys/values = trailing committed-token
    window; per-layer code -> per-layer untied expert table -> [J, E]
    feature."""

    def __init__(self, J: int, E: int, d_tok: int = CTX_D_TOK,
                 d: int = CTX_D, d_out: int = CTX_D_OUT,
                 window: int = CTX_WINDOW):
        super().__init__()
        self.J, self.E = J, E
        self.q = nn.Linear(d_tok, d, bias=False)
        self.k = nn.Linear(d_tok, d, bias=False)
        self.v = nn.Linear(d_tok, d, bias=False)
        self.pos = nn.Embedding(window, d)
        self.proj = nn.Linear(d, J * d_out)
        self.exp_out = nn.Parameter(
            torch.randn(J, E, d_out) / (d_out * 4))
        self.bias = nn.Parameter(torch.zeros(J))   # kills the
        #                             rank-null layer-mean direction
        self.d_out = d_out
        self.scale = d ** -0.5

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, cur: torch.Tensor,
                win: torch.Tensor) -> torch.Tensor:
        """cur [d_tok] current-token features; win [Wc, d_tok]
        trailing window. Returns [J, E]."""
        q = self.q(cur)                                  # [d]
        k = self.k(win) + self.pos.weight[: win.shape[0]]
        att = torch.softmax((k @ q) * self.scale, dim=0)  # [Wc]
        c = att @ self.v(win)                            # [d]
        h = self.proj(c + q).view(self.J, self.d_out)
        return torch.einsum("jd,jed->je", h, self.exp_out) \
            + self.bias[:, None]


class CtxState:
    """Online wrapper: frozen token features + token ring + Adam;
    predict-then-update; MSE on decision rows only."""

    def __init__(self, J: int, E: int, emb: torch.Tensor,
                 dev: torch.device, seed: int = 0,
                 d: int = CTX_D, d_out: int = CTX_D_OUT):
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.model = CtxEncoder(J, E, d=d, d_out=d_out).to(dev)
        torch.random.set_rng_state(g)
        self.emb = emb.to(dev)                 # frozen [V, d_tok]
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=CTX_LR)
        self.ring: list[int] = []
        self.J, self.E, self.dev = J, E, dev

    def reset_seq(self) -> None:
        self.ring.clear()

    def _inputs(self, tok: int):
        cur = self.emb[tok]
        win = self.emb[torch.tensor(self.ring, device=self.dev,
                                    dtype=torch.int64)]
        return cur, win

    def predict(self, tok: int) -> torch.Tensor:
        """[J, E] feature from PRE-update params; zeros with an
        empty window (sequence start)."""
        if not self.ring:
            return torch.zeros(self.J, self.E, device=self.dev)
        with torch.no_grad():
            cur, win = self._inputs(tok)
            return self.model(cur, win)

    def update(self, yb: torch.Tensor, dr: torch.Tensor,
               tok: int) -> None:
        """One Adam step on decision-row MSE (dr [J,E] bool mask,
        e not in prev8), then push tok into the ring."""
        if self.ring:
            n = dr.sum()
            if n > 0:
                cur, win = self._inputs(tok)
                out = self.model(cur, win)
                loss = (((out - yb.float()) ** 2) * dr.float()
                        ).sum() / n.float()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
        self.ring.append(int(tok))
        if len(self.ring) > CTX_WINDOW:
            self.ring.pop(0)


class TinyAttn(nn.Module):
    def __init__(self, n_layers: int, n_experts: int, buckets: int,
                 mode: str = "pooled", window: int = ATTN_WINDOW,
                 d_emb: int = D_EMB, d: int = D_MODEL,
                 zero_gate: bool = False):
        super().__init__()
        if mode not in ("pooled", "expert_query"):
            raise ValueError(f"unknown attn mode {mode!r}")
        self.mode = mode
        # LayerScale output gate: gamma == 0 at init makes the fused
        # challenger EXACTLY the ridge's ranking — parity is the
        # mechanical starting point, training must EARN departure.
        self.gamma = nn.Parameter(torch.zeros(n_layers)) \
            if zero_gate else None
        self.J, self.E, self.W = n_layers, n_experts, window
        self.tok_emb = nn.Embedding(buckets, d_emb)
        self.exp_emb = nn.Embedding(n_experts, d_emb)
        self.layer_emb = nn.Embedding(n_layers, d_emb)
        self.pos_emb = nn.Embedding(window, 8)
        slot_dim = d_emb + d_emb + 8 + 2      # tok, expset, pos, extras
        self.enc = nn.Linear(slot_dim, d)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q = nn.Linear(d_emb, d, bias=False)
        if mode == "pooled":
            self.exp_out = nn.Parameter(torch.randn(n_experts, d) / d)
        else:
            self.u = nn.Linear(d, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = d ** -0.5

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tok_b: torch.Tensor, tops_w: torch.Tensor,
                extras: torch.Tensor) -> torch.Tensor:
        """tok_b [Wc] i64 window token buckets; tops_w [Wc, J, K] i64
        window routed sets; extras [Wc, 2] f32 (mean gate w, conf).
        Returns model_out [J, E] (residual scores). Wc <= W (early
        positions)."""
        Wc = tok_b.shape[0]
        jj = torch.arange(self.J, device=tok_b.device)
        tok = self.tok_emb(tok_b)                          # [Wc, de]
        expset = self.exp_emb(tops_w).mean(dim=2)          # [Wc, J, de]
        pos = self.pos_emb(torch.arange(Wc, device=tok_b.device))
        slots = torch.cat(
            [tok[:, None, :].expand(-1, self.J, -1), expset,
             pos[:, None, :].expand(-1, self.J, -1),
             extras[:, None, :].expand(-1, self.J, -1)],
            dim=2)                                         # [Wc, J, *]
        h = self.enc(slots).permute(1, 0, 2)               # [J, Wc, d]
        keys = self.k(h)
        vals = self.v(h)
        le = self.layer_emb(jj)                            # [J, de]
        if self.mode == "pooled":
            q = self.q(le)[:, None, :]                     # [J, 1, d]
            att = torch.softmax(
                (q @ keys.transpose(1, 2)) * self.scale, dim=2)
            ctx = (att @ vals).squeeze(1)                  # [J, d]
            return ctx @ self.exp_out.t() + self.bias
        # expert_query: q per (layer, expert)
        ee = self.exp_emb(torch.arange(self.E, device=tok_b.device))
        q = self.q(ee[None, :, :] + le[:, None, :])        # [J, E, d]
        att = torch.softmax(
            (q @ keys.transpose(1, 2)) * self.scale, dim=2)  # [J,E,Wc]
        ctx = att @ vals                                   # [J, E, d]
        return self.u(ctx).squeeze(2) + self.bias


class AttnState:
    """Online wrapper: window ring + Adam; predict-then-update."""

    def __init__(self, J: int, E: int, buckets: int, mode: str,
                 dev: torch.device, seed: int = 0,
                 zero_gate: bool = False):
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.model = TinyAttn(J, E, buckets, mode,
                              zero_gate=zero_gate).to(dev)
        torch.random.set_rng_state(g)
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=ATTN_LR)
        self.ring: list[tuple] = []      # (tok_b, tops, extras)
        self.dev = dev

    def reset_seq(self) -> None:
        self.ring.clear()

    def predict(self, ridge_open: torch.Tensor) -> torch.Tensor:
        """Challenger logits [J, E] from the CURRENT window +
        pre-update parameters; ridge_open detached."""
        if not self.ring:
            return ridge_open.detach() * self.model.alpha.detach() \
                + self.model.bias.detach()
        tok_b = torch.stack([r[0] for r in self.ring])
        tops_w = torch.stack([r[1] for r in self.ring])
        extras = torch.stack([r[2] for r in self.ring])
        with torch.no_grad():
            out = self.model(tok_b, tops_w, extras)
            if self.model.gamma is not None:
                out = self.model.gamma[:, None] * out
        return self.model.alpha.detach() * ridge_open.detach() + out

    def update(self, ridge_open: torch.Tensor, yb: torch.Tensor,
               tok_b: int, tops: torch.Tensor,
               extras: tuple[float, float]) -> None:
        """One online Adam step on BCE(challenger logits, yb), then
        push position t into the window ring."""
        if self.ring:
            tok_w = torch.stack([r[0] for r in self.ring])
            tops_w = torch.stack([r[1] for r in self.ring])
            ex_w = torch.stack([r[2] for r in self.ring])
            out = self.model(tok_w, tops_w, ex_w)
            if self.model.gamma is not None:
                out = self.model.gamma[:, None] * out
            logits = self.model.alpha * ridge_open.detach() + out
            loss = F.binary_cross_entropy_with_logits(
                logits, yb.float())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        self.ring.append((
            torch.tensor(tok_b, device=self.dev, dtype=torch.int64),
            tops.clone(),
            torch.tensor(extras, device=self.dev,
                         dtype=torch.float32)))
        if len(self.ring) > ATTN_WINDOW:
            self.ring.pop(0)
