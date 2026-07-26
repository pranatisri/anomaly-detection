"""L2 sequence model: variable-order Markov surprisal over a typed token stream.

The n-gram model is built FIRST and deliberately, not as a fallback afterthought:

  1. It is the sanity check. If a GRU cannot beat an interpolated bigram model, either
     there is a bug or the sequence signal is trivial -- and you want to know which
     before spending an afternoon on CPU training.
  2. It is demo insurance. Pure numpy/dict, fits in seconds, fully deterministic, no
     seed sensitivity, no torch.
  3. "GRU vs n-gram" is one of the few genuinely informative ablation rows in the report.

Events are flattened into ONE typed stream per entity rather than modelled as two
separate sequences (intra-event commands, inter-event transitions). One loss, one model,
and surprisal aggregates cleanly from token to event.

    [BOS] hr:09 res:crm auth:sso ok cmd:login cmd:query cmd:export [EOE] hr:09 res:wiki ...

Namespaced tokens (`res:`, `cmd:`, `auth:`, `hr:`) with NAMESPACE-SPECIFIC unknown
tokens. A single global <UNK> would become a pure attack-indicator: the one token that
only ever appears when something new happens.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Interpolation weights: entity-specific, role/type-level, unigram, uniform floor.
# Held out from any label; these are ordinary smoothing constants.
LAMBDA_ENTITY = 0.40
LAMBDA_GROUP = 0.30
LAMBDA_UNIGRAM = 0.25
LAMBDA_FLOOR = 0.05

MAX_ENTITY_BIGRAMS = 512
VOCAB_FLOOR = 4096.0


def event_tokens(ev: dict) -> List[str]:
    """Flatten one event into typed tokens."""
    toks = ["hr:%02d" % ev["_hour"],
            "res:%s" % ev["resource_accessed"],
            "auth:%s" % ev["auth_method"],
            "ok" if ev["auth_result"] == "success" else "fail"]
    cs = ev.get("command_sequence")
    if cs is not None:
        for c in cs:
            toks.append("cmd:%s" % c)
    return toks


class NGramSurprisal:
    """Interpolated bigram model with per-entity, per-group and global backoff.

    Trained ONLINE on the stream, and only on events the detector has not flagged --
    "unflagged", never "labelled normal", since selecting training data by label is
    itself leakage.
    """

    def __init__(self) -> None:
        self.g_bi: Dict[str, Dict[str, float]] = {}     # group (entity_type) bigrams
        self.g_ctx: Dict[str, float] = {}
        self.uni: Dict[str, float] = {}
        self.uni_total = 0.0
        self.e_bi: Dict[str, Dict[str, float]] = {}     # per-entity, keyed "entity|prev"
        self.e_ctx: Dict[str, float] = {}

    def _p(self, entity: str, group: str, prev: str, tok: str) -> float:
        ek = "%s|%s" % (entity, prev)
        d = self.e_bi.get(ek)
        p_e = (d.get(tok, 0.0) / self.e_ctx[ek]) if d and self.e_ctx.get(ek) else 0.0

        gk = "%s|%s" % (group, prev)
        gd = self.g_bi.get(gk)
        p_g = (gd.get(tok, 0.0) / self.g_ctx[gk]) if gd and self.g_ctx.get(gk) else 0.0

        p_u = (self.uni.get(tok, 0.0) / self.uni_total) if self.uni_total else 0.0
        return (LAMBDA_ENTITY * p_e + LAMBDA_GROUP * p_g
                + LAMBDA_UNIGRAM * p_u + LAMBDA_FLOOR / VOCAB_FLOOR)

    def surprisal(self, entity: str, group: str, toks: List[str]) -> float:
        """Mean bits per token.

        MEAN, not sum. A sum is confounded by how many commands the event happened to
        contain; command count is a real signal but belongs in L1 as its own feature,
        not smuggled inside the sequence score.
        """
        if not toks:
            return 0.0
        prev = "<BOS>"
        total = 0.0
        for t in toks:
            total += -math.log2(max(self._p(entity, group, prev, t), 1e-12))
            prev = t
        return total / len(toks)

    def update(self, entity: str, group: str, toks: List[str]) -> None:
        prev = "<BOS>"
        for t in toks:
            ek = "%s|%s" % (entity, prev)
            d = self.e_bi.get(ek)
            if d is None:
                d = self.e_bi[ek] = {}
            d[t] = d.get(t, 0.0) + 1.0
            self.e_ctx[ek] = self.e_ctx.get(ek, 0.0) + 1.0

            gk = "%s|%s" % (group, prev)
            gd = self.g_bi.get(gk)
            if gd is None:
                gd = self.g_bi[gk] = {}
            gd[t] = gd.get(t, 0.0) + 1.0
            self.g_ctx[gk] = self.g_ctx.get(gk, 0.0) + 1.0

            self.uni[t] = self.uni.get(t, 0.0) + 1.0
            self.uni_total += 1.0
            prev = t

        if len(self.e_bi) > MAX_ENTITY_BIGRAMS * 64:
            self._evict()

    def _evict(self) -> None:
        """Keep state bounded; drop the least-used entity contexts."""
        keep = sorted(self.e_ctx, key=self.e_ctx.get, reverse=True)[: MAX_ENTITY_BIGRAMS * 32]
        ks = set(keep)
        self.e_bi = {k: v for k, v in self.e_bi.items() if k in ks}
        self.e_ctx = {k: v for k, v in self.e_ctx.items() if k in ks}
